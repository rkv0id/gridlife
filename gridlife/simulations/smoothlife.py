import torch
import torch.nn.functional as F

from gridlife.simulations.base import Param, Simulation


class SmoothLife(Simulation):
    name = "smoothlife"
    description = "SmoothLife - Continuous Game of Life"
    channels = 1
    halo_size = 12
    params = {
        "ra": Param(default=12.0, min=4.0, max=12.0, step=1.0, description="Outer radius"),
        "b1": Param(default=0.278, min=0.0, max=1.0, step=0.01, description="Birth lower"),
        "b2": Param(default=0.365, min=0.0, max=1.0, step=0.01, description="Birth upper"),
        "d1": Param(default=0.267, min=0.0, max=1.0, step=0.01, description="Death lower"),
        "d2": Param(default=0.445, min=0.0, max=1.0, step=0.01, description="Death upper"),
        "alpha_n": Param(
            default=0.028, min=0.001, max=0.1, step=0.001, description="Outer sharpness"
        ),
        "alpha_m": Param(
            default=0.147, min=0.001, max=0.5, step=0.001, description="Inner sharpness"
        ),
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
        },
        "gliders": {
            "ra": 12.0,
            "b1": 0.257,
            "b2": 0.336,
            "d1": 0.365,
            "d2": 0.549,
            "alpha_n": 0.028,
            "alpha_m": 0.147,
        },
        "expanding": {
            "ra": 12.0,
            "b1": 0.21,
            "b2": 0.32,
            "d1": 0.26,
            "d2": 0.44,
            "alpha_n": 0.028,
            "alpha_m": 0.147,
        },
    }

    def __init__(self) -> None:
        self._inner_kernel: torch.Tensor | None = None
        self._outer_kernel: torch.Tensor | None = None
        self._cached_ra: float = 0.0

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

        # Anti-aliased inner disk
        inner_mask = (1.0 - (dist - r_inner).clamp(0, 1)).clamp(0, 1)
        inner_sum = inner_mask.sum()
        if inner_sum > 0:
            inner_mask = inner_mask / inner_sum

        # Anti-aliased outer annulus
        outer_ring = (1.0 - (dist - r_outer).clamp(0, 1)).clamp(0, 1)
        outer_mask = outer_ring - inner_mask * inner_sum / (inner_sum if inner_sum > 0 else 1.0)
        # Recompute cleanly: outer annulus = in outer circle but NOT in inner circle
        outer_mask = ((dist > r_inner) & (dist <= r_outer)).float()
        # Smooth the boundary
        boundary = (dist > r_inner - 0.5) & (dist <= r_inner + 0.5)
        outer_mask[boundary] = (dist[boundary] - r_inner + 0.5).clamp(0, 1)
        boundary_out = (dist > r_outer - 0.5) & (dist <= r_outer + 0.5)
        outer_mask[boundary_out] = (r_outer + 0.5 - dist[boundary_out]).clamp(0, 1)

        outer_sum = outer_mask.sum()
        if outer_sum > 0:
            outer_mask = outer_mask / outer_sum

        self._inner_kernel = inner_mask.reshape(1, 1, size, size)
        self._outer_kernel = outer_mask.reshape(1, 1, size, size)
        self._cached_ra = ra
        return self._inner_kernel, self._outer_kernel

    def _sigma(self, x: torch.Tensor, a: float, alpha: float) -> torch.Tensor:
        """Logistic sigmoid: smooth step at threshold a."""
        return 1.0 / (1.0 + torch.exp(-(x - a) / alpha))

    def _sigma_interval(self, x: torch.Tensor, a: float, b: float, alpha: float) -> torch.Tensor:
        """Smooth interval: ~1 when a < x < b, ~0 otherwise."""
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
        """
        SmoothLife transition function s(n, m).
        Discrete time stepping mode: returns the new state directly.
        """
        alive = self._sigma(m, 0.5, alpha_m)
        birth = self._sigma_interval(n, b1, b2, alpha_n)
        death = self._sigma_interval(n, d1, d2, alpha_n)
        return birth * (1.0 - alive) + death * alive

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
        m = F.conv2d(grid4d, inner_k).squeeze(0)  # inner (cell) average
        n = F.conv2d(grid4d, outer_k).squeeze(0)  # outer (neighbor) average

        # Trim extra if kernel radius < halo_size
        trim = h - r
        if trim > 0:
            m = m[:, trim:-trim, trim:-trim]
            n = n[:, trim:-trim, trim:-trim]

        # Discrete time stepping: new state = s(n, m) directly
        result = self._transition(n, m, b1, b2, d1, d2, alpha_n, alpha_m)

        return result.clamp(0, 1)

    def init_grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        grid = torch.zeros(1, height, width, device=device)

        # Sparse small circles - not too many, not too large
        n_circles = max(3, (height * width) // 8000)
        for _ in range(n_circles):
            cy = torch.randint(height // 4, 3 * height // 4, (1,)).item()
            cx = torch.randint(width // 4, 3 * width // 4, (1,)).item()
            r = torch.randint(4, max(5, min(height, width) // 15), (1,)).item()

            y = torch.arange(height, device=device).float() - cy
            x = torch.arange(width, device=device).float() - cx
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            dist = torch.sqrt(xx * xx + yy * yy)

            # Smooth-edged circle
            circle = (1.0 - ((dist - r) / 2.0).clamp(0, 1)).clamp(0, 1)
            grid[0] = (grid[0] + circle).clamp(0, 1)

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
