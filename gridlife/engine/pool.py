from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from gridlife.engine.partition import compute_row_ranges, merge_strips, split_grid
from gridlife.simulations.base import Simulation

if TYPE_CHECKING:
    pass


class LocalWorker:
    """
    In-process worker that owns a horizontal strip of the grid.
    Same logic as StripWorker but no Ray, no serialization, no IPC.
    Direct method calls on plain Python objects.
    """

    def __init__(
        self,
        worker_id: int,
        simulation: Simulation,
        strip_data: torch.Tensor,
        row_offset: int,
    ) -> None:
        self.worker_id = worker_id
        self.simulation = simulation
        self.row_offset = row_offset
        self.halo_size = simulation.halo_size
        self.params = simulation.default_params()
        self.device = torch.device("cpu")

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.owned_height = strip_data.shape[1]
        self.width = strip_data.shape[2]

        h = self.halo_size
        c = strip_data.shape[0]
        self.grid = torch.zeros(c, self.owned_height + 2 * h, self.width, device=self.device)
        self.grid[:, h : h + self.owned_height, :] = strip_data.to(self.device)

        self._last_step_ms: float = 0.0

    def step(self) -> None:
        t0 = time.perf_counter()
        h = self.halo_size
        padded = F.pad(self.grid, (h, h, 0, 0), mode="circular")
        result = self.simulation.step(padded, self.params)
        self.grid[:, h : h + self.owned_height, :] = result
        self._last_step_ms = (time.perf_counter() - t0) * 1000

    def get_top_boundary(self) -> torch.Tensor:
        h = self.halo_size
        return self.grid[:, h : h + h, :]

    def get_bottom_boundary(self) -> torch.Tensor:
        h = self.halo_size
        return self.grid[:, self.owned_height : self.owned_height + h, :]

    def set_top_halo(self, data: torch.Tensor) -> None:
        h = self.halo_size
        self.grid[:, :h, :] = data

    def set_bottom_halo(self, data: torch.Tensor) -> None:
        h = self.halo_size
        self.grid[:, h + self.owned_height :, :] = data

    def render(self) -> bytes:
        h = self.halo_size
        owned = self.grid[:, h : h + self.owned_height, :]
        vis = self.simulation.render_transform(owned)
        indices = (vis * 255).clamp(0, 255).byte()
        return indices.cpu().numpy().tobytes()

    def get_strip_data(self) -> torch.Tensor:
        h = self.halo_size
        return self.grid[:, h : h + self.owned_height, :]

    def set_strip_data(self, data: torch.Tensor, row_offset: int) -> None:
        h = self.halo_size
        self.owned_height = data.shape[1]
        self.row_offset = row_offset
        c = data.shape[0]
        self.grid = torch.zeros(c, self.owned_height + 2 * h, self.width, device=self.device)
        self.grid[:, h : h + self.owned_height, :] = data.to(self.device)

    def update_params(self, params: dict[str, float]) -> None:
        self.params.update(params)

    def perturb(self, row: int, col: int, channel: int, value: float, radius: int) -> None:
        h = self.halo_size
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc <= radius * radius:
                    r = row + dr + h
                    c = col + dc
                    if 0 <= r < self.grid.shape[1] and 0 <= c < self.width:
                        self.grid[channel, r, c] = value


class WorkerPool:
    """
    Abstract interface over a set of strip workers.
    Coordinator uses this without knowing whether workers are
    local objects or Ray actors.
    """

    def __init__(
        self,
        simulation: Simulation,
        width: int,
        height: int,
        num_workers: int,
    ) -> None:
        self.simulation = simulation
        self.width = width
        self.height = height
        self.num_workers = num_workers
        self.row_ranges: list[tuple[int, int]] = []

        self.step_ms: float = 0.0
        self.halo_ms: float = 0.0

    def bootstrap(self, init_grid: torch.Tensor | None = None) -> None:
        raise NotImplementedError

    def do_step(self) -> None:
        raise NotImplementedError

    def collect_grid(self) -> torch.Tensor:
        raise NotImplementedError

    def collect_frame(self) -> list[bytes]:
        raise NotImplementedError

    def repartition(self, new_num_workers: int) -> None:
        raise NotImplementedError

    def update_params(self, params: dict[str, float]) -> None:
        raise NotImplementedError

    def perturb(self, row: int, col: int, channel: int, value: float, radius: int) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


class LocalWorkerPool(WorkerPool):
    """
    In-process worker pool. All workers are plain Python objects.
    Halo exchange is direct tensor slice copies - no serialization,
    no IPC, no Ray. Runs at thousands of steps/sec on a single machine.
    """

    def __init__(
        self,
        simulation: Simulation,
        width: int,
        height: int,
        num_workers: int,
    ) -> None:
        super().__init__(simulation, width, height, num_workers)
        self.workers: list[LocalWorker] = []

    def bootstrap(self, init_grid: torch.Tensor | None = None) -> None:
        if init_grid is not None:
            grid = init_grid
        else:
            grid = self.simulation.init_grid(self.height, self.width, torch.device("cpu"))

        self.row_ranges = compute_row_ranges(self.height, self.num_workers)
        strips = split_grid(grid, self.num_workers)

        self.workers = []
        for i, (strip, (row_start, _)) in enumerate(zip(strips, self.row_ranges, strict=True)):
            self.workers.append(LocalWorker(i, self.simulation, strip, row_start))

    def do_step(self) -> None:
        n = len(self.workers)

        # Halo exchange first
        t0 = time.perf_counter()
        tops = [w.get_top_boundary() for w in self.workers]
        bots = [w.get_bottom_boundary() for w in self.workers]
        for i in range(n):
            above = (i - 1) % n
            below = (i + 1) % n
            self.workers[i].set_top_halo(bots[above])
            self.workers[i].set_bottom_halo(tops[below])
        self.halo_ms = (time.perf_counter() - t0) * 1000

        # Then step all workers
        t0 = time.perf_counter()
        for w in self.workers:
            w.step()
        self.step_ms = (time.perf_counter() - t0) * 1000

    def collect_grid(self) -> torch.Tensor:
        strips = [w.get_strip_data() for w in self.workers]
        return merge_strips(strips)

    def collect_frame(self) -> list[bytes]:
        return [w.render() for w in self.workers]

    def repartition(self, new_num_workers: int) -> None:
        grid = self.collect_grid()
        self.num_workers = new_num_workers
        self.row_ranges = compute_row_ranges(self.height, new_num_workers)
        strips = split_grid(grid, new_num_workers)

        self.workers = []
        for i, (strip, (row_start, _)) in enumerate(zip(strips, self.row_ranges, strict=True)):
            self.workers.append(LocalWorker(i, self.simulation, strip, row_start))

    def update_params(self, params: dict[str, float]) -> None:
        for w in self.workers:
            w.update_params(params)

    def perturb(self, row: int, col: int, channel: int, value: float, radius: int) -> None:
        for i, (start, end) in enumerate(self.row_ranges):
            if start <= row < end:
                self.workers[i].perturb(row - start, col, channel, value, radius)
                return

    def shutdown(self) -> None:
        self.workers.clear()
