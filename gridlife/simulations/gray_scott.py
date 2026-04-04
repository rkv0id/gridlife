import torch
import torch.nn.functional as F

from gridlife.simulations.base import Param, Simulation


class GrayScott(Simulation):
    name = "gray_scott"
    description = "Gray-Scott Reaction-Diffusion"
    channels = 2
    halo_size = 1
    params = {
        "feed_rate": Param(
            default=0.0367, min=0.01, max=0.1, step=0.001, description="Feed rate (F)"
        ),
        "kill_rate": Param(
            default=0.0649, min=0.01, max=0.1, step=0.001, description="Kill rate (k)"
        ),
        "diffusion_u": Param(default=0.21, min=0.0, max=0.3, step=0.01, description="U diffusion"),
        "diffusion_v": Param(default=0.105, min=0.0, max=0.3, step=0.01, description="V diffusion"),
        "dt": Param(default=1.0, min=0.1, max=2.0, step=0.1, description="Time step"),
    }
    presets = {
        "mitosis": {"feed_rate": 0.0367, "kill_rate": 0.0649},
        "coral": {"feed_rate": 0.0545, "kill_rate": 0.062},
        "spirals": {"feed_rate": 0.014, "kill_rate": 0.045},
        "worms": {"feed_rate": 0.078, "kill_rate": 0.061},
        "holes": {"feed_rate": 0.039, "kill_rate": 0.058},
    }

    def __init__(self) -> None:
        self._laplacian = torch.tensor(
            [[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]],
            dtype=torch.float32,
        ).reshape(1, 1, 3, 3)

    def step(self, grid: torch.Tensor, params: dict[str, float]) -> torch.Tensor:
        f = params["feed_rate"]
        k = params["kill_rate"]
        du = params["diffusion_u"]
        dv = params["diffusion_v"]
        dt = params["dt"]

        lap_kernel = self._laplacian.to(grid.device)

        u_padded = grid[0:1].unsqueeze(0)
        v_padded = grid[1:2].unsqueeze(0)

        lap_u = F.conv2d(u_padded, lap_kernel).squeeze(0)
        lap_v = F.conv2d(v_padded, lap_kernel).squeeze(0)

        u = grid[0:1, 1:-1, 1:-1]
        v = grid[1:2, 1:-1, 1:-1]

        uvv = u * v * v
        du_dt = du * lap_u - uvv + f * (1.0 - u)
        dv_dt = dv * lap_v + uvv - (f + k) * v

        u_new = (u + dt * du_dt).clamp(0, 1)
        v_new = (v + dt * dv_dt).clamp(0, 1)

        return torch.cat([u_new, v_new], dim=0)

    def init_grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        grid = torch.zeros(2, height, width, device=device)
        grid[0] = 1.0

        # Seed covers a significant portion of the grid for visible patterns
        cy, cx = height // 2, width // 2
        size = max(10, min(height, width) // 6)
        y0, y1 = cy - size, cy + size
        x0, x1 = cx - size, cx + size
        grid[1, y0:y1, x0:x1] = 0.5
        grid[1, y0:y1, x0:x1] += torch.rand(y1 - y0, x1 - x0, device=device) * 0.1
        grid[1].clamp_(0, 1)

        return grid

    def render_channel(self) -> int:
        return 1

    def palette(self) -> torch.Tensor:
        pal = torch.zeros(256, 3, dtype=torch.uint8)
        for i in range(256):
            t = i / 255.0
            if t < 0.25:
                s = t / 0.25
                pal[i] = torch.tensor([int(s * 60), 0, int(s * 100)], dtype=torch.uint8)
            elif t < 0.5:
                s = (t - 0.25) / 0.25
                pal[i] = torch.tensor([60 + int(s * 140), 0, 100 + int(s * 20)], dtype=torch.uint8)
            elif t < 0.75:
                s = (t - 0.5) / 0.25
                pal[i] = torch.tensor(
                    [200 + int(s * 55), int(s * 130), int((1 - s) * 120)], dtype=torch.uint8
                )
            else:
                s = (t - 0.75) / 0.25
                pal[i] = torch.tensor([255, 130 + int(s * 125), int(s * 200)], dtype=torch.uint8)
        return pal
