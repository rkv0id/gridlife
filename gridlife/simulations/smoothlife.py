import torch
import torch.nn.functional as F

from gridlife.simulations.base import Param, Simulation

# Fixed step size for Euler time integration (SmoothLifeL uses dt~0.05-0.1).
EULER_DT = 0.1


class SmoothLife(Simulation):
    """
    SmoothLife (Rafler 2011, arXiv:1111.1567).

    Two rule families. "discrete" uses direct replacement A' = s(n, m) per
    the original paper. "euler" uses smooth time stepping
    A' = A + dt * (s(n, m) - A), required for the SmoothLifeL ruleset that
    produces the dense wicks-and-wires regime.
    """

    name = "smoothlife"
    description = "SmoothLife - Continuous Game of Life"
    channels = 1
    halo_size = 12
    preset_only = True
    params = {
        "ra": Param(default=12.0, min=4.0, max=12.0, step=1.0, description="Outer radius"),
        "b1": Param(default=0.278, min=0.0, max=1.0, step=0.01, description="Birth lower"),
        "b2": Param(default=0.365, min=0.0, max=1.0, step=0.01, description="Birth upper"),
        "d1": Param(default=0.267, min=0.0, max=1.0, step=0.01, description="Survive lower"),
        "d2": Param(default=0.445, min=0.0, max=1.0, step=0.01, description="Survive upper"),
        "alpha_n": Param(
            default=0.028, min=0.001, max=0.1, step=0.001, description="Outer sharpness"
        ),
        "alpha_m": Param(
            default=0.147, min=0.001, max=0.5, step=0.001, description="Inner sharpness"
        ),
    }
    presets = {
        # Rafler paper classic values, discrete time stepping. Stable blobs.
        "stable": {
            "ra": 12.0,
            "b1": 0.278,
            "b2": 0.365,
            "d1": 0.267,
            "d2": 0.445,
            "alpha_n": 0.028,
            "alpha_m": 0.147,
            "_mode": "discrete",
        },
        # SmoothLifeL with Euler integration. Dense wicks-and-wires regime.
        # Individual gliders occasionally emerge within but are not isolated.
        "wicks": {
            "ra": 12.0,
            "b1": 0.257,
            "b2": 0.336,
            "d1": 0.365,
            "d2": 0.549,
            "alpha_n": 0.028,
            "alpha_m": 0.147,
            "_mode": "euler",
        },
    }

    def __init__(self) -> None:
        self._inner_kernel: torch.Tensor | None = None
        self._outer_kernel: torch.Tensor | None = None
        self._cached_ra: float = 0.0
        self._mode: str = "discrete"

    def apply_preset_metadata(self, preset: dict[str, float | str]) -> None:
        mode = preset.get("_mode")
        if isinstance(mode, str):
            self._mode = mode

    def _build_kernels(self, ra: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
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

        inner_mask = (1.0 - (dist - r_inner + 0.5).clamp(0, 1)).clamp(0, 1)
        inner_sum = inner_mask.sum()
        if inner_sum > 0:
            inner_mask = inner_mask / inner_sum

        outer_full = (1.0 - (dist - r_outer + 0.5).clamp(0, 1)).clamp(0, 1)
        outer_inner = (1.0 - (dist - r_inner + 0.5).clamp(0, 1)).clamp(0, 1)
        outer_mask = (outer_full - outer_inner).clamp(0, 1)

        outer_sum = outer_mask.sum()
        if outer_sum > 0:
            outer_mask = outer_mask / outer_sum

        self._inner_kernel = inner_mask.reshape(1, 1, size, size)
        self._outer_kernel = outer_mask.reshape(1, 1, size, size)
        self._cached_ra = ra
        return self._inner_kernel, self._outer_kernel

    def _sigma(self, x: torch.Tensor, a: float, alpha: float) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-(x - a) / alpha))

    def _sigma_interval(self, x: torch.Tensor, a: float, b: float, alpha: float) -> torch.Tensor:
        return self._sigma(x, a, alpha) * (1.0 - self._sigma(x, b, alpha))

    def _transition(
        self,
        n: torch.Tensor,
        m: torch.Tensor,
        b1: float,
        b2: float,
        d1: float,
        d2: float,
        alpha_n: float,
        alpha_m: float,
    ) -> torch.Tensor:
        """Rafler s(n, m) = birth * (1-alive) + survive * alive."""
        alive = self._sigma(m, 0.5, alpha_m)
        birth = self._sigma_interval(n, b1, b2, alpha_n)
        survive = self._sigma_interval(n, d1, d2, alpha_n)
        return birth * (1.0 - alive) + survive * alive

    def step(self, grid: torch.Tensor, params: dict[str, float]) -> torch.Tensor:
        ra = params["ra"]
        b1 = params["b1"]
        b2 = params["b2"]
        d1 = params["d1"]
        d2 = params["d2"]
        alpha_n = params["alpha_n"]
        alpha_m = params["alpha_m"]

        r = int(ra)
        h = self.halo_size
        inner_k, outer_k = self._build_kernels(ra, grid.device)

        grid4d = grid.unsqueeze(0)
        m = F.conv2d(grid4d, inner_k).squeeze(0)
        n = F.conv2d(grid4d, outer_k).squeeze(0)

        trim = h - r
        if trim > 0:
            m = m[:, trim:-trim, trim:-trim]
            n = n[:, trim:-trim, trim:-trim]

        s = self._transition(n, m, b1, b2, d1, d2, alpha_n, alpha_m)

        if self._mode == "euler":
            inner = grid[:, h:-h, h:-h]
            return (inner + EULER_DT * (s - inner)).clamp(0, 1)
        return s.clamp(0, 1)

    def init_grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        grid = torch.zeros(1, height, width, device=device)
        n_patches = max(5, (height * width) // 5000)
        for _ in range(n_patches):
            cy = int(torch.randint(height // 4, 3 * height // 4, (1,)).item())
            cx = int(torch.randint(width // 4, 3 * width // 4, (1,)).item())
            r = int(torch.randint(8, 20, (1,)).item())

            y = torch.arange(height, device=device).float() - cy
            x = torch.arange(width, device=device).float() - cx
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            dist = torch.sqrt(xx * xx + yy * yy)

            mask = (dist < r).float()
            patch = mask * (0.3 + 0.7 * torch.rand(height, width, device=device))
            grid[0] = (grid[0] + patch).clamp(0, 1)

        return grid

    def palette(self) -> torch.Tensor:
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
