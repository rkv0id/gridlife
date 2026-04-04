import torch
import torch.nn.functional as F

from gridlife.simulations.base import Param, Simulation


class Lenia(Simulation):
    name = "lenia"
    description = "Lenia - Continuous Cellular Automata"
    channels = 1
    halo_size = 13
    params = {
        "R": Param(default=13.0, min=5.0, max=13.0, step=1.0, description="Kernel radius"),
        "T": Param(default=10.0, min=1.0, max=20.0, step=1.0, description="Time step divisor"),
        "mu": Param(default=0.15, min=0.0, max=0.5, step=0.01, description="Growth center"),
        "sigma": Param(default=0.017, min=0.001, max=0.1, step=0.001, description="Growth width"),
    }
    presets = {
        "orbium": {"R": 13.0, "T": 10.0, "mu": 0.15, "sigma": 0.017},
        "geminium": {"R": 10.0, "T": 10.0, "mu": 0.14, "sigma": 0.014},
        "smooth_blob": {"R": 13.0, "T": 10.0, "mu": 0.12, "sigma": 0.02},
        "pulsing": {"R": 13.0, "T": 10.0, "mu": 0.21, "sigma": 0.03},
    }

    def __init__(self) -> None:
        self._kernel: torch.Tensor | None = None
        self._kernel_r: float = 0.0

    def _build_kernel(self, R: float, device: torch.device) -> torch.Tensor:
        """Build the ring-shaped Lenia kernel. Cached until R changes."""
        if self._kernel is not None and self._kernel_r == R and self._kernel.device == device:
            return self._kernel

        r = int(R)
        size = 2 * r + 1
        y = torch.arange(size, device=device).float() - r
        x = torch.arange(size, device=device).float() - r
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dist = torch.sqrt(xx * xx + yy * yy) / r

        kernel = torch.zeros(size, size, device=device)
        mask = (dist > 0) & (dist < 1)
        d = dist[mask]
        kernel[mask] = torch.exp(4.0 - 4.0 / (4.0 * d * (1.0 - d)))

        kernel = kernel / kernel.sum()

        self._kernel = kernel.reshape(1, 1, size, size)
        self._kernel_r = R
        return self._kernel

    def step(self, grid: torch.Tensor, params: dict[str, float]) -> torch.Tensor:
        R = params["R"]
        T = params["T"]
        mu = params["mu"]
        sigma = params["sigma"]
        r = int(R)
        h = self.halo_size

        kernel = self._build_kernel(R, grid.device)

        # grid: (1, H+2h, W+2h) where h = halo_size (max radius)
        # conv2d with kernel (2r+1) strips r on each side
        potential = F.conv2d(grid.unsqueeze(0), kernel).squeeze(0)

        # potential is (1, H+2h-2r, W+2h-2r)
        # We need the center (1, H, W) region
        trim = h - r
        if trim > 0:
            potential = potential[:, trim:-trim, trim:-trim]

        growth = 2.0 * torch.exp(-((potential - mu) ** 2) / (2.0 * sigma * sigma)) - 1.0

        # Inner owned region: strip halo from input
        inner = grid[:, h:-h, h:-h]

        result = (inner + (1.0 / T) * growth).clamp(0, 1)

        return result

    def init_grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        grid = torch.zeros(1, height, width, device=device)

        cy, cx = height // 2, width // 2
        radius = max(10, min(height, width) // 6)

        y = torch.arange(height, device=device).float() - cy
        x = torch.arange(width, device=device).float() - cx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dist = torch.sqrt(xx * xx + yy * yy)

        blob = torch.exp(-((dist / (radius * 0.5)) ** 2))
        blob = blob * (0.5 + 0.5 * torch.rand(height, width, device=device))
        grid[0] = blob.clamp(0, 1)

        return grid

    def palette(self) -> torch.Tensor:
        pal = torch.zeros(256, 3, dtype=torch.uint8)
        for i in range(256):
            t = i / 255.0
            if t < 0.2:
                s = t / 0.2
                pal[i] = torch.tensor([int(s * 40), int(s * 20), int(s * 10)], dtype=torch.uint8)
            elif t < 0.5:
                s = (t - 0.2) / 0.3
                pal[i] = torch.tensor(
                    [40 + int(s * 160), 20 + int(s * 80), 10 + int(s * 20)], dtype=torch.uint8
                )
            elif t < 0.8:
                s = (t - 0.5) / 0.3
                pal[i] = torch.tensor(
                    [200 + int(s * 55), 100 + int(s * 120), 30 + int(s * 50)], dtype=torch.uint8
                )
            else:
                s = (t - 0.8) / 0.2
                pal[i] = torch.tensor(
                    [255, 220 + int(s * 35), 80 + int(s * 175)], dtype=torch.uint8
                )
        return pal
