import torch

from gridlife.engine.pool import LocalWorkerPool, WorkerPool
from gridlife.simulations.base import Simulation


class Coordinator:
    """
    Orchestrates distributed simulation via a WorkerPool.
    Doesn't know or care whether workers are local objects or Ray actors.
    """

    def __init__(
        self,
        simulation: Simulation,
        width: int,
        height: int,
        num_workers: int,
        gpu: bool = False,
        init_grid: torch.Tensor | None = None,
        pool: WorkerPool | None = None,
    ) -> None:
        self.simulation = simulation
        self.width = width
        self.height = height
        self.num_workers = num_workers
        self.step_count = 0

        if pool is not None:
            self.pool = pool
        else:
            self.pool = LocalWorkerPool(simulation, width, height, num_workers)

        self.pool.bootstrap(init_grid)

    @property
    def row_ranges(self) -> list[tuple[int, int]]:
        return self.pool.row_ranges

    def do_step(self) -> None:
        self.pool.do_step()
        self.step_count += 1

    def run_steps(self, n: int) -> None:
        for _ in range(n):
            self.do_step()

    def collect_grid(self) -> torch.Tensor:
        return self.pool.collect_grid()

    def collect_frame(self) -> list[bytes]:
        return self.pool.collect_frame()

    def repartition(self, new_num_workers: int) -> None:
        self.pool.repartition(new_num_workers)
        self.num_workers = new_num_workers

    def update_params(self, params: dict[str, float]) -> None:
        self.pool.update_params(params)

    def perturb(self, row: int, col: int, channel: int, value: float, radius: int) -> None:
        self.pool.perturb(row, col, channel, value, radius)

    def shutdown(self) -> None:
        self.pool.shutdown()

    def get_metrics(self) -> dict[str, object]:
        return {
            "step_count": self.step_count,
            "step_ms": self.pool.step_ms,
            "halo_ms": self.pool.halo_ms,
            "num_workers": self.num_workers,
        }
