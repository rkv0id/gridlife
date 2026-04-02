import torch
import torch.nn.functional as F

from gridlife.simulations.base import Simulation


class GameOfLife(Simulation):
    name = "game_of_life"
    description = "Conway's Game of Life"
    channels = 1
    halo_size = 1
    params = {}
    pixelated = True

    def __init__(self) -> None:
        # 3x3 kernel that counts all 8 neighbors (center = 0)
        self._kernel = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32).reshape(
            1, 1, 3, 3
        )

    def step(self, grid: torch.Tensor, params: dict[str, float]) -> torch.Tensor:
        kernel = self._kernel.to(grid.device)
        # grid: (1, H+2, W+2) - has 1-cell halo on all sides
        # conv2d with no padding strips the halo, giving (1, H, W)
        neighbors = F.conv2d(grid, kernel)
        alive = grid[:, 1:-1, 1:-1]
        birth = (alive < 0.5) & (neighbors == 3)
        survive = (alive > 0.5) & ((neighbors == 2) | (neighbors == 3))
        return (birth | survive).float()

    def init_grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        return (torch.rand(1, height, width, device=device) < 0.25).float()

    def palette(self) -> torch.Tensor:
        pal = torch.zeros(256, 3, dtype=torch.uint8)
        pal[128:] = torch.tensor([0, 255, 0], dtype=torch.uint8)
        return pal
