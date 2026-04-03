from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import ray
import torch

from gridlife.engine.partition import compute_row_ranges, merge_strips, split_grid
from gridlife.engine.pool import WorkerPool
from gridlife.engine.worker import StripWorker
from gridlife.simulations.base import Simulation

if TYPE_CHECKING:
    pass


class RayWorkerPool(WorkerPool):
    """
    Ray actor-based worker pool for distributed cluster execution.
    Workers are Ray actors that run on remote machines.
    """

    def __init__(
        self,
        simulation: Simulation,
        width: int,
        height: int,
        num_workers: int,
        gpu: bool = False,
    ) -> None:
        super().__init__(simulation, width, height, num_workers)
        self.gpu = gpu
        self.workers: list[Any] = []

    def bootstrap(self, init_grid: torch.Tensor | None = None) -> None:
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
        n = len(self.workers)

        # Halo exchange first
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
        self.halo_ms = (time.perf_counter() - t0) * 1000

        # Then step
        t0 = time.perf_counter()
        ray.get([w.step.remote() for w in self.workers])
        self.step_ms = (time.perf_counter() - t0) * 1000

    def collect_grid(self) -> torch.Tensor:
        strips = ray.get([w.get_strip_data.remote() for w in self.workers])
        return merge_strips(strips)

    def collect_frame(self) -> list[bytes]:
        return ray.get([w.render.remote() for w in self.workers])

    def repartition(self, new_num_workers: int) -> None:
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
        ray.get([w.update_params.remote(params) for w in self.workers])

    def perturb(self, row: int, col: int, channel: int, value: float, radius: int) -> None:
        for i, (start, end) in enumerate(self.row_ranges):
            if start <= row < end:
                ray.get(self.workers[i].perturb.remote(row - start, col, channel, value, radius))
                return

    def shutdown(self) -> None:
        for w in self.workers:
            ray.kill(w)
        self.workers.clear()

    def kill_worker(self, worker_id: int | None = None) -> int | None:
        if len(self.workers) <= 1:
            return None

        import random

        if worker_id is None:
            worker_id = random.randint(0, len(self.workers) - 1)

        if worker_id < 0 or worker_id >= len(self.workers):
            return None

        # Collect all strips, zero-fill the dead one
        strips = []
        for i, w in enumerate(self.workers):
            if i == worker_id:
                h: torch.Tensor = ray.get(w.get_strip_data.remote())
                strips.append(torch.zeros_like(h))
                ray.kill(w)
            else:
                strips.append(ray.get(w.get_strip_data.remote()))

        grid = merge_strips(strips)

        # Clear remaining workers
        for i, w in enumerate(self.workers):
            if i != worker_id:
                ray.kill(w)
        self.workers.clear()

        # Repartition to N-1
        new_count = len(strips) - 1
        self.num_workers = new_count
        self.row_ranges = compute_row_ranges(self.height, new_count)
        new_strips = split_grid(grid, new_count)

        for i, (strip, (row_start, _)) in enumerate(zip(new_strips, self.row_ranges, strict=True)):
            if self.gpu:
                worker = StripWorker.options(num_gpus=1).remote(
                    i, self.simulation, strip, row_start
                )
            else:
                worker = StripWorker.options(num_cpus=1).remote(
                    i, self.simulation, strip, row_start
                )
            self.workers.append(worker)

        return worker_id

    def heal_worker(self) -> None:
        self.repartition(len(self.workers) + 1)
