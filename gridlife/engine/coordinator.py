import time
from typing import Any

import ray
import torch

from gridlife.engine.partition import compute_row_ranges, merge_strips, split_grid
from gridlife.engine.worker import StripWorker
from gridlife.simulations.base import Simulation


class Coordinator:
    """Orchestrates distributed simulation. Runs in the caller's process, not as a Ray actor."""

    def __init__(
        self,
        simulation: Simulation,
        width: int,
        height: int,
        num_workers: int,
        gpu: bool = False,
        init_grid: torch.Tensor | None = None,
    ) -> None:
        self.simulation = simulation
        self.width = width
        self.height = height
        self.num_workers = num_workers
        self.gpu = gpu

        self.workers: list[Any] = []
        self.row_ranges: list[tuple[int, int]] = []
        self.step_count = 0
        self.running = False

        self._step_ms: float = 0.0
        self._halo_ms: float = 0.0

        self._bootstrap(init_grid)

    def _bootstrap(self, init_grid: torch.Tensor | None = None) -> None:
        """Create workers, partition grid, distribute initial state."""
        if init_grid is not None:
            grid = init_grid
        else:
            grid = self.simulation.init_grid(self.height, self.width, torch.device("cpu"))

        self.row_ranges = compute_row_ranges(self.height, self.num_workers)
        strips = split_grid(grid, self.num_workers)

        self.workers = []
        for i, (strip, (row_start, _)) in enumerate(zip(strips, self.row_ranges, strict=True)):
            if self.gpu:
                worker = StripWorker.options(num_gpus=1).remote(
                    i, self.simulation, strip, row_start
                )
            else:
                worker = StripWorker.options(num_cpus=1).remote(
                    i, self.simulation, strip, row_start
                )
            self.workers.append(worker)

    def do_step(self) -> None:
        """Execute one simulation step across all workers."""
        n = len(self.workers)

        # Halo exchange FIRST (workers need valid halos before stepping)
        t0 = time.perf_counter()

        top_futs = [w.get_top_boundary.remote() for w in self.workers]
        bot_futs = [w.get_bottom_boundary.remote() for w in self.workers]
        tops = ray.get(top_futs)
        bots = ray.get(bot_futs)

        halo_futs = []
        for i in range(n):
            above = (i - 1) % n
            below = (i + 1) % n
            halo_futs.append(self.workers[i].set_top_halo.remote(bots[above]))
            halo_futs.append(self.workers[i].set_bottom_halo.remote(tops[below]))
        ray.get(halo_futs)

        self._halo_ms = (time.perf_counter() - t0) * 1000

        # THEN all workers compute their step in parallel
        t0 = time.perf_counter()
        ray.get([w.step.remote() for w in self.workers])
        self._step_ms = (time.perf_counter() - t0) * 1000

        self.step_count += 1

    def run_steps(self, n: int) -> None:
        """Run n steps synchronously."""
        for _ in range(n):
            self.do_step()

    def collect_grid(self) -> torch.Tensor:
        """Collect full grid from all workers."""
        strips = ray.get([w.get_strip_data.remote() for w in self.workers])
        return merge_strips(strips)

    def repartition(self, new_num_workers: int) -> None:
        """Rescale to a different number of workers."""
        grid = self.collect_grid()

        for w in self.workers:
            ray.kill(w)
        self.workers.clear()

        self.num_workers = new_num_workers
        self.row_ranges = compute_row_ranges(self.height, new_num_workers)
        strips = split_grid(grid, new_num_workers)

        self.workers = []
        for i, (strip, (row_start, _)) in enumerate(zip(strips, self.row_ranges, strict=True)):
            if self.gpu:
                worker = StripWorker.options(num_gpus=1).remote(
                    i, self.simulation, strip, row_start
                )
            else:
                worker = StripWorker.options(num_cpus=1).remote(
                    i, self.simulation, strip, row_start
                )
            self.workers.append(worker)

    def update_params(self, params: dict[str, float]) -> None:
        """Push parameter update to all workers."""
        ray.get([w.update_params.remote(params) for w in self.workers])

    def shutdown(self) -> None:
        """Kill all workers."""
        for w in self.workers:
            ray.kill(w)
        self.workers.clear()

    def get_metrics(self) -> dict[str, object]:
        return {
            "step_count": self.step_count,
            "step_ms": self._step_ms,
            "halo_ms": self._halo_ms,
            "num_workers": len(self.workers),
        }
