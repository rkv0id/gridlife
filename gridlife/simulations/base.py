from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Param:
    """A tunable simulation parameter exposed to the UI."""

    default: float
    min: float
    max: float
    step: float = 0.001
    description: str = ""


class Simulation:
    """Base class for grid simulations. Subclass and override step() and init_grid()."""

    name: str = "Unnamed"
    description: str = ""
    channels: int = 1
    halo_size: int = 1
    params: dict[str, Param] = {}
    presets: dict[str, dict[str, float | str]] = {}

    # If True, UI hides sliders and only shows preset selector.
    # Changing presets resets the simulation.
    preset_only: bool = False

    def step(self, grid: torch.Tensor, params: dict[str, float]) -> torch.Tensor:
        """
        Compute one simulation step.

        grid: (channels, H + 2*halo, W + 2*halo) with valid halo data on all sides.
        params: current parameter values.

        Returns: (channels, H, W) - the updated inner region, halos stripped.
        """
        raise NotImplementedError

    def init_grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """
        Create initial grid state.

        Returns: (channels, height, width) tensor.
        """
        raise NotImplementedError

    def palette(self) -> torch.Tensor:
        """(256, 3) uint8 colormap. Grid values in [0,1] map to indices 0-255."""
        ramp = torch.arange(256, dtype=torch.uint8)
        return torch.stack([ramp, ramp, ramp], dim=1)

    def render_channel(self) -> int:
        return 0

    def render_transform(self, grid: torch.Tensor) -> torch.Tensor:
        """(channels, H, W) -> (H, W) with values in [0, 1]."""
        return grid[self.render_channel()].clamp(0, 1)

    def default_params(self) -> dict[str, float]:
        return {k: v.default for k, v in self.params.items()}
