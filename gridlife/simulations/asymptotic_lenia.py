import torch
import torch.nn.functional as F

from gridlife.simulations.base import Param, Simulation


class AsymptoticLenia(Simulation):
    """
    Asymptotic Lenia (Kawaguchi et al. 2021).

    Update rule: A_t+dt = A_t + dt * (T(K*A) - A_t), where T(u) = (G(u)+1)/2.
    Single-ring kernel with Orbium-neighborhood parameters sits in ALenia's
    Turing-stripe basin, producing dense emergent wave patterns from random
    initial conditions rather than isolated gliders.
    """

    name = "asymptotic_lenia"
    description = "Asymptotic Lenia - Smooth Continuous CA"
    channels = 1
    halo_size = 13
    preset_only = True
    params = {
        "R": Param(default=13.0, min=5.0, max=13.0, step=1.0, description="Kernel radius"),
        "dt": Param(default=0.1, min=0.01, max=0.3, step=0.01, description="Time step"),
        "mu": Param(default=0.15, min=0.0, max=0.5, step=0.01, description="Target center"),
        "sigma": Param(default=0.015, min=0.001, max=0.1, step=0.001, description="Target width"),
    }
    presets = {
        "turing": {"R": 13.0, "dt": 0.1, "mu": 0.15, "sigma": 0.015},
        "stripes": {"R": 13.0, "dt": 0.1, "mu": 0.16, "sigma": 0.017},
    }

    def __init__(self) -> None:
        self._kernel: torch.Tensor | None = None
        self._kernel_r: float = 0.0

    def _build_kernel(self, R: float, device: torch.device) -> torch.Tensor:
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
        dt = params["dt"]
        mu = params["mu"]
        sigma = params["sigma"]
        r = int(R)
        h = self.halo_size

        kernel = self._build_kernel(R, grid.device)
        potential = F.conv2d(grid.unsqueeze(0), kernel).squeeze(0)

        trim = h - r
        if trim > 0:
            potential = potential[:, trim:-trim, trim:-trim]

        # T(u) = (G(u)+1)/2 = exp(-(u-mu)^2 / (2*sigma^2))
        target = torch.exp(-((potential - mu) ** 2) / (2.0 * sigma * sigma))
        inner = grid[:, h:-h, h:-h]
        return (inner + dt * (target - inner)).clamp(0, 1)

    def init_grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        grid = torch.zeros(1, height, width, device=device)

        n_patches = max(3, (height * width) // 10000)
        for _ in range(n_patches):
            cy = int(torch.randint(height // 4, 3 * height // 4, (1,)).item())
            cx = int(torch.randint(width // 4, 3 * width // 4, (1,)).item())
            patch_r = 20
            y0, y1 = max(0, cy - patch_r), min(height, cy + patch_r)
            x0, x1 = max(0, cx - patch_r), min(width, cx + patch_r)
            grid[0, y0:y1, x0:x1] = torch.rand(y1 - y0, x1 - x0, device=device)

        return grid

    def palette(self) -> torch.Tensor:
        pal = torch.zeros(256, 3, dtype=torch.uint8)
        for i in range(256):
            t = i / 255.0
            if t < 0.25:
                s = t / 0.25
                pal[i] = torch.tensor([int(s * 30), 0, int(s * 60)], dtype=torch.uint8)
            elif t < 0.55:
                s = (t - 0.25) / 0.3
                pal[i] = torch.tensor(
                    [30 + int(s * 100), int(s * 20), 60 + int(s * 120)], dtype=torch.uint8
                )
            elif t < 0.8:
                s = (t - 0.55) / 0.25
                pal[i] = torch.tensor(
                    [130 + int(s * 90), 20 + int(s * 60), 180 + int(s * 40)], dtype=torch.uint8
                )
            else:
                s = (t - 0.8) / 0.2
                pal[i] = torch.tensor(
                    [220 + int(s * 35), 80 + int(s * 175), 220 + int(s * 35)], dtype=torch.uint8
                )
        return pal
