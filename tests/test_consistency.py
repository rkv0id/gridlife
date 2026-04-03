import torch
import torch.nn.functional as F

from gridlife.engine.partition import merge_strips, split_grid
from gridlife.simulations.base import Simulation
from gridlife.simulations.game_of_life import GameOfLife
from gridlife.simulations.gray_scott import GrayScott
from gridlife.simulations.lenia import Lenia
from gridlife.simulations.smoothlife import SmoothLife


def run_single(sim: Simulation, grid: torch.Tensor, steps: int) -> torch.Tensor:
    """Run simulation on a single grid for N steps. The baseline."""
    params = sim.default_params()
    h = sim.halo_size
    for _ in range(steps):
        padded = F.pad(grid, (h, h, h, h), mode="circular")
        grid = sim.step(padded, params)
    return grid


def run_multi_strip(
    sim: Simulation, grid: torch.Tensor, num_workers: int, steps: int
) -> torch.Tensor:
    """
    Run simulation split across N strips with manual halo exchange.
    Simulates what the distributed engine will do, without Ray.
    """
    params = sim.default_params()
    h = sim.halo_size
    strips = split_grid(grid, num_workers)

    for _ in range(steps):
        # Build halos: each strip needs h rows from its neighbors (toroidal)
        padded_strips: list[torch.Tensor] = []
        for i in range(num_workers):
            above = (i - 1) % num_workers
            below = (i + 1) % num_workers

            # Vertical halos from neighbors
            top_halo = strips[above][:, -h:, :]
            bottom_halo = strips[below][:, :h, :]

            # Stack: top_halo + owned + bottom_halo
            with_vertical = torch.cat([top_halo, strips[i], bottom_halo], dim=1)

            # Horizontal halos (circular wrap within the strip)
            padded = F.pad(with_vertical, (h, h, 0, 0), mode="circular")
            padded_strips.append(padded)

        # Step each strip independently
        new_strips: list[torch.Tensor] = []
        for padded in padded_strips:
            result = sim.step(padded, params)
            new_strips.append(result)

        strips = new_strips

    return merge_strips(strips)


class TestMultiWorkerConsistency:
    """
    The most important test in the project.
    Verifies that splitting across N workers produces identical results
    to running on a single grid.
    """

    def _check_consistency(
        self, sim: Simulation, height: int, width: int, steps: int, workers: list[int]
    ) -> None:
        torch.manual_seed(42)
        grid = sim.init_grid(height, width, torch.device("cpu"))

        baseline = run_single(sim, grid.clone(), steps)

        for n in workers:
            result = run_multi_strip(sim, grid.clone(), n, steps)
            assert torch.allclose(baseline, result, atol=1e-6), (
                f"{sim.name}: {n} workers diverged from baseline after {steps} steps, "
                f"max diff = {(baseline - result).abs().max().item()}"
            )

    def test_gol_even_split(self) -> None:
        self._check_consistency(GameOfLife(), 64, 64, 50, [2, 4, 8])

    def test_gol_uneven_split(self) -> None:
        self._check_consistency(GameOfLife(), 67, 64, 50, [3, 5, 7])

    def test_gol_single_vs_multi(self) -> None:
        self._check_consistency(GameOfLife(), 32, 32, 100, [1, 2])

    def test_gray_scott_even_split(self) -> None:
        self._check_consistency(GrayScott(), 64, 64, 30, [2, 4])

    def test_gray_scott_uneven_split(self) -> None:
        self._check_consistency(GrayScott(), 67, 64, 30, [3, 5])

    def test_gray_scott_many_workers(self) -> None:
        """Stress test: more workers than makes sense, small strips."""
        self._check_consistency(GrayScott(), 32, 32, 20, [8, 16])

    def test_single_row_strips(self) -> None:
        """Edge case: each worker gets 1-2 rows."""
        self._check_consistency(GameOfLife(), 8, 16, 20, [4, 8])

    def test_lenia_even_split(self) -> None:
        sim = Lenia()
        # Lenia has halo_size=13, needs height > 2*13 per worker
        self._check_consistency(sim, 64, 64, 10, [2])

    def test_smoothlife_even_split(self) -> None:
        sim = SmoothLife()
        self._check_consistency(sim, 64, 64, 10, [2])
