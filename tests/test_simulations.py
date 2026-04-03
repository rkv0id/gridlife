import torch
import torch.nn.functional as F

from gridlife.simulations.base import Simulation
from gridlife.simulations.game_of_life import GameOfLife
from gridlife.simulations.gray_scott import GrayScott
from gridlife.simulations.lenia import Lenia
from gridlife.simulations.smoothlife import SmoothLife


def pad_and_step(
    sim: Simulation,
    grid: torch.Tensor,
    params: dict[str, float] | None = None,
) -> torch.Tensor:
    """Helper: pad grid with circular halos, run one step."""
    h = sim.halo_size
    padded = F.pad(grid, (h, h, h, h), mode="circular")
    if params is None:
        params = sim.default_params()
    return sim.step(padded, params)


class TestGameOfLife:
    def setup_method(self) -> None:
        self.sim = GameOfLife()

    def test_block_still_life(self) -> None:
        """A 2x2 block should remain unchanged."""
        grid = torch.zeros(1, 6, 6)
        grid[0, 2, 2] = 1
        grid[0, 2, 3] = 1
        grid[0, 3, 2] = 1
        grid[0, 3, 3] = 1

        result = pad_and_step(self.sim, grid)
        assert torch.equal(result, grid)

    def test_blinker_oscillator(self) -> None:
        """Horizontal blinker becomes vertical after one step."""
        grid = torch.zeros(1, 6, 6)
        grid[0, 2, 1] = 1
        grid[0, 2, 2] = 1
        grid[0, 2, 3] = 1

        result = pad_and_step(self.sim, grid)

        expected = torch.zeros(1, 6, 6)
        expected[0, 1, 2] = 1
        expected[0, 2, 2] = 1
        expected[0, 3, 2] = 1

        assert torch.equal(result, expected)

    def test_blinker_period_2(self) -> None:
        """Blinker returns to original after 2 steps."""
        grid = torch.zeros(1, 6, 6)
        grid[0, 2, 1] = 1
        grid[0, 2, 2] = 1
        grid[0, 2, 3] = 1
        original = grid.clone()

        step1 = pad_and_step(self.sim, grid)
        step2 = pad_and_step(self.sim, step1)

        assert torch.equal(step2, original)

    def test_dead_grid_stays_dead(self) -> None:
        grid = torch.zeros(1, 8, 8)
        result = pad_and_step(self.sim, grid)
        assert torch.equal(result, grid)

    def test_overcrowding(self) -> None:
        """Center cell of a 3x3 block has 8 neighbors - dies."""
        grid = torch.zeros(1, 6, 6)
        grid[0, 1:4, 1:4] = 1

        result = pad_and_step(self.sim, grid)
        assert result[0, 2, 2] == 0

    def test_init_grid_density(self) -> None:
        torch.manual_seed(42)
        grid = self.sim.init_grid(100, 100, torch.device("cpu"))
        density = grid.mean().item()
        assert 0.15 < density < 0.35

    def test_palette_shape(self) -> None:
        pal = self.sim.palette()
        assert pal.shape == (256, 3)
        assert pal.dtype == torch.uint8


class TestGrayScott:
    def setup_method(self) -> None:
        self.sim = GrayScott()

    def test_step_preserves_shape(self) -> None:
        grid = self.sim.init_grid(32, 32, torch.device("cpu"))
        result = pad_and_step(self.sim, grid)
        assert result.shape == grid.shape

    def test_step_values_bounded(self) -> None:
        """Values stay in [0, 1] after many steps."""
        grid = self.sim.init_grid(32, 32, torch.device("cpu"))
        for _ in range(50):
            grid = pad_and_step(self.sim, grid)
        assert grid.min() >= 0.0
        assert grid.max() <= 1.0

    def test_uniform_u_stable(self) -> None:
        """U=1, V=0 everywhere should remain stable."""
        grid = torch.zeros(2, 16, 16)
        grid[0] = 1.0

        result = pad_and_step(self.sim, grid)
        assert torch.allclose(result[0], torch.ones(16, 16), atol=1e-5)
        assert torch.allclose(result[1], torch.zeros(16, 16), atol=1e-5)

    def test_init_grid_channels(self) -> None:
        grid = self.sim.init_grid(64, 64, torch.device("cpu"))
        assert grid.shape == (2, 64, 64)
        assert grid[0].mean() > 0.9
        assert grid[1].mean() < 0.1

    def test_presets_have_valid_keys(self) -> None:
        valid_keys = set(self.sim.params.keys())
        for preset_name, preset_vals in self.sim.presets.items():
            for k in preset_vals:
                assert k in valid_keys, f"Preset '{preset_name}' has invalid key '{k}'"

    def test_palette_shape(self) -> None:
        pal = self.sim.palette()
        assert pal.shape == (256, 3)
        assert pal.dtype == torch.uint8


class TestLenia:
    def setup_method(self) -> None:
        self.sim = Lenia()

    def test_step_preserves_shape(self) -> None:
        grid = self.sim.init_grid(64, 64, torch.device("cpu"))
        result = pad_and_step(self.sim, grid)
        assert result.shape == grid.shape

    def test_step_values_bounded(self) -> None:
        grid = self.sim.init_grid(64, 64, torch.device("cpu"))
        for _ in range(20):
            grid = pad_and_step(self.sim, grid)
        assert grid.min() >= 0.0
        assert grid.max() <= 1.0

    def test_palette_shape(self) -> None:
        pal = self.sim.palette()
        assert pal.shape == (256, 3)
        assert pal.dtype == torch.uint8


class TestSmoothLife:
    def setup_method(self) -> None:
        self.sim = SmoothLife()

    def test_step_preserves_shape(self) -> None:
        grid = self.sim.init_grid(64, 64, torch.device("cpu"))
        result = pad_and_step(self.sim, grid)
        assert result.shape == grid.shape

    def test_step_values_bounded(self) -> None:
        grid = self.sim.init_grid(64, 64, torch.device("cpu"))
        for _ in range(20):
            grid = pad_and_step(self.sim, grid)
        assert grid.min() >= 0.0
        assert grid.max() <= 1.0

    def test_palette_shape(self) -> None:
        pal = self.sim.palette()
        assert pal.shape == (256, 3)
        assert pal.dtype == torch.uint8
