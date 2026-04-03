import torch
import torch.nn.functional as F

from gridlife.simulations.base import Param, Simulation


class SmoothLife(Simulation):
    name = "smoothlife"
    description = "SmoothLife - Continuous Game of Life"
    channels = 1
    halo_size = 12
    params = {
        "ra": Param(default=12.0, min=4.0, max=20.0, step=1.0, description="Outer radius"),
        "b1": Param(default=0.278, min=0.0, max=1.0, step=0.01, description="Birth lower"),
        "b2": Param(default=0.365, min=0.0, max=1.0, step=0.01, description="Birth upper"),
        "d1": Param(default=0.267, min=0.0, max=1.0, step=0.01, description="Death lower"),
        "d2": Param(default=0.445, min=0.0, max=1.0, step=0.01, description="Death upper"),
        "alpha_n": Param(
            default=0.028, min=0.001, max=0.1, step=0.001, description="Outer transition sharpness"
        ),
        "alpha_m": Param(
            default=0.147, min=0.001, max=0.5, step=0.001, description="Inner transition sharpness"
        ),
        "dt": Param(default=0.1, min=0.01, max=0.5, step=0.01, description="Time step"),
    }
    presets = {
        "default": {
            "ra": 12.0,
            "b1": 0.278,
            "b2": 0.365,
            "d1": 0.267,
            "d2": 0.445,
            "alpha_n": 0.028,
            "alpha_m": 0.147,
            "dt": 0.1,
        },
        "gliders": {
            "ra": 12.0,
            "b1": 0.257,
            "b2": 0.336,
            "d1": 0.365,
            "d2": 0.549,
            "alpha_n": 0.028,
            "alpha_m": 0.147,
            "dt": 0.1,
        },
        "ribbons": {
            "ra": 10.0,
            "b1": 0.241,
            "b2": 0.343,
            "d1": 0.340,
            "d2": 0.518,
            "alpha_n": 0.028,
            "alpha_m": 0.147,
            "dt": 0.12,
        },
    }

    def __init__(self) -> None:
        self._inner_kernel: torch.Tensor | None = None
        self._outer_kernel: torch.Tensor | None = None
        self._cached_ra: float = 0.0

    def _build_kernels(self, ra: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Build inner disk and outer annulus kernels."""
        if (
            self._inner_kernel is not None
            and self._outer_kernel is not None
            and self._cached_ra == ra
            and self._inner_kernel.device == device
        ):
            return self._inner_kernel, self._outer_kernel

        r_outer = int(ra)
        r_inner = max(1, int(ra / 3.0))
        size = 2 * r_outer + 1

        y = torch.arange(size, device=device).float() - r_outer
        x = torch.arange(size, device=device).float() - r_outer
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dist = torch.sqrt(xx * xx + yy * yy)

        # Inner disk kernel (for cell state average)
        inner_mask = (dist <= r_inner).float()
        inner_sum = inner_mask.sum()
        if inner_sum > 0:
            inner_mask = inner_mask / inner_sum

        # Outer annulus kernel (for neighborhood average)
        outer_mask = ((dist > r_inner) & (dist <= r_outer)).float()
        outer_sum = outer_mask.sum()
        if outer_sum > 0:
            outer_mask = outer_mask / outer_sum

        self._inner_kernel = inner_mask.reshape(1, 1, size, size)
        self._outer_kernel = outer_mask.reshape(1, 1, size, size)
        self._cached_ra = ra
        return self._inner_kernel, self._outer_kernel

    def _sigma(self, x: torch.Tensor, a: float, alpha: float) -> torch.Tensor:
        """Smooth step function (logistic sigmoid)."""
        return 1.0 / (1.0 + torch.exp(-(x - a) / alpha))

    def _sigma_interval(self, x: torch.Tensor, a: float, b: float, alpha: float) -> torch.Tensor:
        """Smooth interval function: high when a < x < b."""
        return self._sigma(x, a, alpha) * (1.0 - self._sigma(x, b, alpha))

    def step(self, grid: torch.Tensor, params: dict[str, float]) -> torch.Tensor:
        ra = params["ra"]
        b1 = params["b1"]
        b2 = params["b2"]
        d1 = params["d1"]
        d2 = params["d2"]
        alpha_n = params["alpha_n"]
        alpha_m = params["alpha_m"]
        dt = params["dt"]

        r = int(ra)
        inner_k, outer_k = self._build_kernels(ra, grid.device)

        # grid: (1, H+2r, W+2r) with halos
        # conv2d with kernel size (2r+1) strips the halos
        m = F.conv2d(grid, inner_k)  # inner neighborhood mean
        n = F.conv2d(grid, outer_k)  # outer neighborhood mean

        # Smooth transition function
        # Birth: cell is dead (m low), comes alive if n in [b1, b2]
        # Death: cell is alive (m high), stays alive if n in [d1, d2]
        alive = self._sigma(m, 0.5, alpha_m)
        birth = self._sigma_interval(n, b1, b2, alpha_n)
        death = self._sigma_interval(n, d1, d2, alpha_n)

        # Interpolate between birth and death based on aliveness
        transition = birth * (1.0 - alive) + death * alive

        # Extract inner region from input
        inner = grid[:, r : grid.shape[1] - r, r : grid.shape[2] - r]

        # Update
        result = (inner + dt * (2.0 * transition - 1.0)).clamp(0, 1)

        return result

    def init_grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        # Random smooth blobs
        grid = torch.zeros(1, height, width, device=device)

        # Several random circular blobs
        n_blobs = max(5, (height * width) // 5000)
        for _ in range(n_blobs):
            cy = torch.randint(0, height, (1,)).item()
            cx = torch.randint(0, width, (1,)).item()
            r = torch.randint(5, max(6, min(height, width) // 8), (1,)).item()

            y = torch.arange(height, device=device).float() - cy
            x = torch.arange(width, device=device).float() - cx
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            dist = torch.sqrt(xx * xx + yy * yy)

            blob = torch.exp(-((dist / (r * 0.4)) ** 2))
            grid[0] = (grid[0] + blob).clamp(0, 1)

        return grid

    def palette(self) -> torch.Tensor:
        # Cool blue-teal gradient
        pal = torch.zeros(256, 3, dtype=torch.uint8)
        for i in range(256):
            t = i / 255.0
            if t < 0.3:
                s = t / 0.3
                pal[i] = torch.tensor([0, int(s * 30), int(s * 60)], dtype=torch.uint8)
            elif t < 0.6:
                s = (t - 0.3) / 0.3
                pal[i] = torch.tensor(
                    [int(s * 20), 30 + int(s * 100), 60 + int(s * 100)], dtype=torch.uint8
                )
            elif t < 0.85:
                s = (t - 0.6) / 0.25
                pal[i] = torch.tensor(
                    [20 + int(s * 80), 130 + int(s * 80), 160 + int(s * 60)], dtype=torch.uint8
                )
            else:
                s = (t - 0.85) / 0.15
                pal[i] = torch.tensor(
                    [100 + int(s * 155), 210 + int(s * 45), 220 + int(s * 35)], dtype=torch.uint8
                )
        return pal
