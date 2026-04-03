import torch
import torch.nn.functional as F

from gridlife.engine.coordinator import Coordinator
from gridlife.simulations.base import Simulation
from gridlife.simulations.game_of_life import GameOfLife
from gridlife.simulations.gray_scott import GrayScott


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


def run_coordinator(
    sim: Simulation, height: int, width: int, steps: int, num_workers: int, seed: int
) -> torch.Tensor:
    torch.manual_seed(seed)
    grid = sim.init_grid(height, width, torch.device("cpu"))
    coord = Coordinator(sim, width, height, num_workers, init_grid=grid)
    coord.run_steps(steps)
    result = coord.collect_grid()
    coord.shutdown()
    return result


class TestLocalPoolConsistency:
    """LocalWorkerPool must match single-grid baseline exactly."""

    def test_gol_2_workers(self) -> None:
        sim = GameOfLife()
        baseline = run_local_baseline(sim, 64, 64, 50, seed=42)
        result = run_coordinator(sim, 64, 64, 50, 2, seed=42)
        assert torch.allclose(baseline, result, atol=1e-6)

    def test_gol_4_workers(self) -> None:
        sim = GameOfLife()
        baseline = run_local_baseline(sim, 64, 64, 50, seed=42)
        result = run_coordinator(sim, 64, 64, 50, 4, seed=42)
        assert torch.allclose(baseline, result, atol=1e-6)

    def test_gol_uneven(self) -> None:
        sim = GameOfLife()
        baseline = run_local_baseline(sim, 67, 64, 30, seed=42)
        result = run_coordinator(sim, 67, 64, 30, 3, seed=42)
        assert torch.allclose(baseline, result, atol=1e-6)

    def test_gray_scott_2_workers(self) -> None:
        sim = GrayScott()
        baseline = run_local_baseline(sim, 64, 64, 30, seed=42)
        result = run_coordinator(sim, 64, 64, 30, 2, seed=42)
        assert torch.allclose(baseline, result, atol=1e-6)

    def test_gray_scott_4_workers(self) -> None:
        sim = GrayScott()
        baseline = run_local_baseline(sim, 64, 64, 30, seed=42)
        result = run_coordinator(sim, 64, 64, 30, 4, seed=42)
        assert torch.allclose(baseline, result, atol=1e-6)


class TestKillHeal:
    """Worker kill/heal must not crash and must preserve grid structure."""

    def test_kill_reduces_workers(self) -> None:
        sim = GameOfLife()
        coord = Coordinator(sim, 64, 64, 4)
        coord.run_steps(10)
        assert coord.num_workers == 4

        killed = coord.kill_worker()
        assert killed is not None
        assert coord.num_workers == 3
        coord.shutdown()

    def test_heal_adds_worker(self) -> None:
        sim = GameOfLife()
        coord = Coordinator(sim, 64, 64, 3)
        coord.run_steps(10)

        coord.heal_worker()
        assert coord.num_workers == 4
        coord.shutdown()

    def test_kill_then_heal_restores_count(self) -> None:
        sim = GameOfLife()
        coord = Coordinator(sim, 64, 64, 4)
        coord.run_steps(10)

        coord.kill_worker()
        assert coord.num_workers == 3

        coord.heal_worker()
        assert coord.num_workers == 4
        coord.shutdown()

    def test_kill_preserves_grid_shape(self) -> None:
        sim = GameOfLife()
        coord = Coordinator(sim, 64, 64, 4)
        coord.run_steps(10)

        coord.kill_worker()
        grid = coord.collect_grid()
        assert grid.shape == (1, 64, 64)
        coord.shutdown()

    def test_simulation_continues_after_kill(self) -> None:
        """Simulation should keep running after a worker is killed."""
        sim = GameOfLife()
        coord = Coordinator(sim, 64, 64, 4)
        coord.run_steps(10)

        coord.kill_worker()
        coord.run_steps(10)

        grid = coord.collect_grid()
        assert grid.shape == (1, 64, 64)
        assert ((grid == 0) | (grid == 1)).all()
        coord.shutdown()

    def test_kill_heal_cycle(self) -> None:
        """Multiple kill/heal cycles should not corrupt state."""
        sim = GameOfLife()
        coord = Coordinator(sim, 64, 64, 4)
        coord.run_steps(10)

        for _ in range(3):
            coord.kill_worker()
            coord.run_steps(5)
            coord.heal_worker()
            coord.run_steps(5)

        grid = coord.collect_grid()
        assert grid.shape == (1, 64, 64)
        assert coord.num_workers == 4
        coord.shutdown()

    def test_cannot_kill_last_worker(self) -> None:
        sim = GameOfLife()
        coord = Coordinator(sim, 64, 64, 1)
        result = coord.kill_worker()
        assert result is None
        assert coord.num_workers == 1
        coord.shutdown()

    def test_kill_specific_worker(self) -> None:
        sim = GameOfLife()
        coord = Coordinator(sim, 64, 64, 4)
        coord.run_steps(10)

        killed = coord.kill_worker(worker_id=2)
        assert killed == 2
        assert coord.num_workers == 3
        coord.shutdown()

    def test_repartition_after_kill(self) -> None:
        """Repartitioning after kill should work normally."""
        sim = GameOfLife()
        coord = Coordinator(sim, 64, 64, 4)
        coord.run_steps(10)

        coord.kill_worker()
        assert coord.num_workers == 3

        coord.repartition(5)
        assert coord.num_workers == 5

        grid = coord.collect_grid()
        assert grid.shape == (1, 64, 64)
        coord.shutdown()
