import torch
import torch.nn.functional as F

from gridlife.simulations.asymptotic_lenia import AsymptoticLenia
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

        result1 = pad_and_step(self.sim, grid)
        result2 = pad_and_step(self.sim, result1)
        assert torch.equal(result2, grid)


class TestGrayScott:
    def setup_method(self) -> None:
        self.sim = GrayScott()

    def test_step_preserves_shape(self) -> None:
        grid = self.sim.init_grid(32, 32, torch.device("cpu"))
        result = pad_and_step(self.sim, grid)
        assert result.shape == grid.shape

    def test_step_values_bounded(self) -> None:
        grid = self.sim.init_grid(32, 32, torch.device("cpu"))
        for _ in range(20):
            grid = pad_and_step(self.sim, grid)
        assert grid.min() >= 0.0
        assert grid.max() <= 1.0

    def test_uniform_zero_v_stays_zero(self) -> None:
        """With no V, U should stay at 1 (feed replenishes)."""
        grid = torch.ones(2, 16, 16) * torch.tensor([[1.0], [0.0]]).reshape(2, 1, 1)
        grid[0] = 1.0
        grid[1] = 0.0

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
                if k.startswith("_"):
                    continue
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

    def test_orbium_init(self) -> None:
        """Default init places an Orbium creature."""
        grid = self.sim.init_grid(64, 64, torch.device("cpu"))
        assert grid.sum() > 0
        assert grid.max() > 0.5

    def test_orbium_pair_init(self) -> None:
        self.sim.apply_preset_metadata({"_init": "orbium_pair"})
        grid = self.sim.init_grid(128, 128, torch.device("cpu"))
        assert grid.sum() > 0
        assert grid.max() > 0.5

    def test_palette_shape(self) -> None:
        pal = self.sim.palette()
        assert pal.shape == (256, 3)
        assert pal.dtype == torch.uint8

    def test_preset_only_flag(self) -> None:
        assert self.sim.preset_only is True


class TestAsymptoticLenia:
    def setup_method(self) -> None:
        self.sim = AsymptoticLenia()

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

    def test_random_init_nonempty(self) -> None:
        grid = self.sim.init_grid(64, 64, torch.device("cpu"))
        assert grid.sum() > 0

    def test_palette_shape(self) -> None:
        pal = self.sim.palette()
        assert pal.shape == (256, 3)
        assert pal.dtype == torch.uint8

    def test_preset_only_flag(self) -> None:
        assert self.sim.preset_only is True


class TestSmoothLife:
    def setup_method(self) -> None:
        self.sim = SmoothLife()

    def test_step_preserves_shape(self) -> None:
        grid = self.sim.init_grid(64, 64, torch.device("cpu"))
        result = pad_and_step(self.sim, grid)
        assert result.shape == grid.shape

    def test_step_values_bounded_discrete(self) -> None:
        grid = self.sim.init_grid(64, 64, torch.device("cpu"))
        for _ in range(20):
            grid = pad_and_step(self.sim, grid)
        assert grid.min() >= 0.0
        assert grid.max() <= 1.0

    def test_step_values_bounded_euler(self) -> None:
        self.sim.apply_preset_metadata({"_mode": "euler"})
        grid = self.sim.init_grid(64, 64, torch.device("cpu"))
        for _ in range(20):
            grid = pad_and_step(self.sim, grid)
        assert grid.min() >= 0.0
        assert grid.max() <= 1.0

    def test_palette_shape(self) -> None:
        pal = self.sim.palette()
        assert pal.shape == (256, 3)
        assert pal.dtype == torch.uint8

    def test_preset_only_flag(self) -> None:
        assert self.sim.preset_only is True
