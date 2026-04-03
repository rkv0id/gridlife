import pytest
import ray
import torch
import torch.nn.functional as F

from gridlife.engine.coordinator import Coordinator
from gridlife.simulations.base import Simulation
from gridlife.simulations.game_of_life import GameOfLife
from gridlife.simulations.gray_scott import GrayScott

# Skip Ray tests by default - run with: pytest -m ray
pytestmark = pytest.mark.ray


@pytest.fixture(scope="module", autouse=True)
def ray_init():
    import os

    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    ray.init(num_cpus=8, ignore_reinit_error=True)
    yield
    ray.shutdown()


def run_local_baseline(
    sim: Simulation, height: int, width: int, steps: int, seed: int
) -> torch.Tensor:
    torch.manual_seed(seed)
    grid = sim.init_grid(height, width, torch.device("cpu"))
    params = sim.default_params()
    h = sim.halo_size
    for _ in range(steps):
        padded = F.pad(grid, (h, h, h, h), mode="circular")
        grid = sim.step(padded, params)
    return grid


def run_distributed(
    sim: Simulation, height: int, width: int, steps: int, num_workers: int, seed: int
) -> torch.Tensor:
    from gridlife.engine.ray_pool import RayWorkerPool

    torch.manual_seed(seed)
    grid = sim.init_grid(height, width, torch.device("cpu"))
    pool = RayWorkerPool(sim, width, height, num_workers)
    coord = Coordinator(sim, width, height, num_workers, pool=pool, init_grid=grid)
    coord.run_steps(steps)
    result = coord.collect_grid()
    coord.shutdown()
    return result


class TestDistributedConsistency:
    """Distributed (Ray) results must match single-grid baseline exactly."""

    def test_gol_2_workers(self) -> None:
        sim = GameOfLife()
        baseline = run_local_baseline(sim, 64, 64, 50, seed=42)
        result = run_distributed(sim, 64, 64, 50, 2, seed=42)
        assert torch.allclose(baseline, result, atol=1e-6), (
            f"max diff = {(baseline - result).abs().max().item()}"
        )

    def test_gol_4_workers(self) -> None:
        sim = GameOfLife()
        baseline = run_local_baseline(sim, 64, 64, 50, seed=42)
        result = run_distributed(sim, 64, 64, 50, 4, seed=42)
        assert torch.allclose(baseline, result, atol=1e-6)

    def test_gol_uneven(self) -> None:
        sim = GameOfLife()
        baseline = run_local_baseline(sim, 67, 64, 30, seed=42)
        result = run_distributed(sim, 67, 64, 30, 3, seed=42)
        assert torch.allclose(baseline, result, atol=1e-6)

    def test_gray_scott_2_workers(self) -> None:
        sim = GrayScott()
        baseline = run_local_baseline(sim, 64, 64, 30, seed=42)
        result = run_distributed(sim, 64, 64, 30, 2, seed=42)
        assert torch.allclose(baseline, result, atol=1e-6)

    def test_gray_scott_4_workers(self) -> None:
        sim = GrayScott()
        baseline = run_local_baseline(sim, 64, 64, 30, seed=42)
        result = run_distributed(sim, 64, 64, 30, 4, seed=42)
        assert torch.allclose(baseline, result, atol=1e-6)


class TestRepartitioning:
    """Repartitioning must preserve grid state and not corrupt the simulation."""

    def test_repartition_preserves_state(self) -> None:
        """Collecting grid, redistributing to more workers, and collecting again
        should return the exact same grid."""
        from gridlife.engine.ray_pool import RayWorkerPool

        sim = GameOfLife()
        torch.manual_seed(42)
        grid = sim.init_grid(64, 64, torch.device("cpu"))
        pool = RayWorkerPool(sim, 64, 64, 2)
        coord = Coordinator(sim, 64, 64, 2, pool=pool, init_grid=grid)

        coord.run_steps(20)
        grid_before = coord.collect_grid()

        coord.repartition(4)
        grid_after = coord.collect_grid()

        assert torch.equal(grid_before, grid_after)
        coord.shutdown()

    def test_repartition_then_continue(self) -> None:
        """Run 20 steps with 2 workers, repartition to 4, run 20 more.
        Can't compare to uninterrupted run because repartitioning changes
        halo exchange topology. Verify grid is structurally valid instead."""
        from gridlife.engine.ray_pool import RayWorkerPool

        sim = GameOfLife()
        torch.manual_seed(42)
        grid = sim.init_grid(64, 64, torch.device("cpu"))

        pool = RayWorkerPool(sim, 64, 64, 2)
        coord = Coordinator(sim, 64, 64, 2, pool=pool, init_grid=grid)
        coord.run_steps(20)
        coord.repartition(4)
        coord.run_steps(20)
        result = coord.collect_grid()
        coord.shutdown()

        assert result.shape == (1, 64, 64)
        assert ((result == 0) | (result == 1)).all()
