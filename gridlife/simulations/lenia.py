import torch
import torch.nn.functional as F

from gridlife.simulations.base import Param, Simulation

# Orbium bicaudatus from Bert Chan's Lenia Jupyter notebook.
# Canonical Lenia glider (20x20), paired with R=13, T=10, mu=0.15, sigma=0.014.
ORBIUM_CELLS: list[list[float]] = [
    [0, 0, 0, 0, 0, 0, 0.1, 0.14, 0.1, 0, 0, 0.03, 0.03, 0, 0, 0.3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0.08, 0.24, 0.3, 0.3, 0.18, 0.14, 0.15, 0.16, 0.15, 0.09, 0.2, 0, 0, 0, 0],
    [
        0,
        0,
        0,
        0,
        0,
        0.15,
        0.34,
        0.44,
        0.46,
        0.38,
        0.18,
        0.14,
        0.11,
        0.13,
        0.19,
        0.18,
        0.45,
        0,
        0,
        0,
    ],
    [0, 0, 0, 0, 0.06, 0.13, 0.39, 0.5, 0.5, 0.37, 0.06, 0, 0, 0, 0.02, 0.16, 0.68, 0, 0, 0],
    [0, 0, 0, 0.11, 0.17, 0.17, 0.33, 0.4, 0.38, 0.28, 0.14, 0, 0, 0, 0, 0, 0.18, 0.42, 0, 0],
    [0, 0, 0.09, 0.18, 0.13, 0.06, 0.08, 0.26, 0.32, 0.32, 0.27, 0, 0, 0, 0, 0, 0, 0.82, 0, 0],
    [0.27, 0, 0.16, 0.12, 0, 0, 0, 0.25, 0.38, 0.44, 0.45, 0.34, 0, 0, 0, 0, 0, 0.22, 0.17, 0],
    [0, 0.07, 0.2, 0.02, 0, 0, 0, 0.31, 0.48, 0.57, 0.6, 0.57, 0, 0, 0, 0, 0, 0, 0.49, 0],
    [0, 0.59, 0.19, 0, 0, 0, 0, 0.2, 0.57, 0.69, 0.76, 0.76, 0.49, 0, 0, 0, 0, 0, 0.36, 0],
    [0, 0.58, 0.19, 0, 0, 0, 0, 0, 0.67, 0.83, 0.9, 0.92, 0.87, 0.12, 0, 0, 0, 0, 0.22, 0.07],
    [0, 0, 0.46, 0, 0, 0, 0, 0, 0.7, 0.93, 1, 1, 1, 0.61, 0, 0, 0, 0, 0.18, 0.11],
    [0, 0, 0.82, 0, 0, 0, 0, 0, 0.47, 1, 1, 0.98, 1, 0.96, 0.27, 0, 0, 0, 0.19, 0.1],
    [0, 0, 0.46, 0, 0, 0, 0, 0, 0.25, 1, 1, 0.84, 0.92, 0.97, 0.54, 0.14, 0.04, 0.1, 0.21, 0.05],
    [0, 0, 0, 0.4, 0, 0, 0, 0, 0.09, 0.95, 1, 0.78, 0.8, 0.83, 0.45, 0.15, 0.17, 0.21, 0.09, 0],
    [0, 0, 0, 0.36, 0.1, 0, 0, 0, 0.05, 0.89, 0.98, 0.77, 0.68, 0.48, 0.06, 0, 0.16, 0.19, 0, 0],
    [0, 0, 0, 0.01, 0.3, 0.07, 0, 0, 0.08, 0.65, 0.92, 0.74, 0.45, 0.16, 0, 0, 0.12, 0.21, 0, 0],
    [
        0,
        0,
        0,
        0,
        0.1,
        0.24,
        0.14,
        0.1,
        0.15,
        0.34,
        0.71,
        0.71,
        0.28,
        0,
        0,
        0.03,
        0.15,
        0.12,
        0.02,
        0,
    ],
    [0, 0, 0, 0, 0, 0.08, 0.21, 0.21, 0.22, 0.17, 0.36, 0.53, 0.15, 0, 0, 0.07, 0.17, 0.07, 0, 0],
    [0, 0, 0, 0, 0, 0, 0.03, 0.13, 0.19, 0.22, 0.19, 0.2, 0.05, 0, 0, 0.09, 0.14, 0.04, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.06, 0.08, 0.09, 0.07, 0.02, 0, 0.01, 0.07, 0.03, 0, 0],
]


def place_orbium(grid: torch.Tensor, cy: int, cx: int, device: torch.device) -> None:
    creature = torch.tensor(ORBIUM_CELLS, dtype=torch.float32, device=device)
    ch, cw = creature.shape
    y0 = cy - ch // 2
    x0 = cx - cw // 2
    y1 = y0 + ch
    x1 = x0 + cw
    height, width = grid.shape[1], grid.shape[2]
    if y0 >= 0 and y1 <= height and x0 >= 0 and x1 <= width:
        grid[0, y0:y1, x0:x1] = creature


class Lenia(Simulation):
    name = "lenia"
    description = "Lenia - Continuous Cellular Automata"
    channels = 1
    halo_size = 13
    preset_only = True
    params = {
        "R": Param(default=13.0, min=5.0, max=13.0, step=1.0, description="Kernel radius"),
        "T": Param(default=10.0, min=1.0, max=20.0, step=1.0, description="Time step divisor"),
        "mu": Param(default=0.15, min=0.0, max=0.5, step=0.01, description="Growth center"),
        "sigma": Param(default=0.014, min=0.001, max=0.1, step=0.001, description="Growth width"),
    }
    presets = {
        "orbium": {"R": 13.0, "T": 10.0, "mu": 0.15, "sigma": 0.014, "_init": "orbium"},
        "orbium_swarm": {"R": 13.0, "T": 10.0, "mu": 0.15, "sigma": 0.014, "_init": "orbium_swarm"},
    }

    def __init__(self) -> None:
        self._kernel: torch.Tensor | None = None
        self._kernel_r: float = 0.0
        self._init_mode: str = "orbium"

    def apply_preset_metadata(self, preset: dict[str, float | str]) -> None:
        init_mode = preset.get("_init")
        if isinstance(init_mode, str):
            self._init_mode = init_mode

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
        T = params["T"]
        mu = params["mu"]
        sigma = params["sigma"]
        r = int(R)
        h = self.halo_size

        kernel = self._build_kernel(R, grid.device)
        potential = F.conv2d(grid.unsqueeze(0), kernel).squeeze(0)

        trim = h - r
        if trim > 0:
            potential = potential[:, trim:-trim, trim:-trim]

        growth = 2.0 * torch.exp(-((potential - mu) ** 2) / (2.0 * sigma * sigma)) - 1.0
        inner = grid[:, h:-h, h:-h]
        return (inner + (1.0 / T) * growth).clamp(0, 1)

    def init_grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        grid = torch.zeros(1, height, width, device=device)

        if self._init_mode == "orbium":
            place_orbium(grid, height // 2, width // 2, device)
        elif self._init_mode == "orbium_swarm":
            positions = [
                (height // 3, width // 3),
                (height // 3, 2 * width // 3),
                (2 * height // 3, width // 3),
                (2 * height // 3, 2 * width // 3),
                (height // 2, width // 2),
            ]
            for cy, cx in positions:
                place_orbium(grid, cy, cx, device)

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
