import time

import ray
import torch
import torch.nn.functional as F

from gridlife.simulations.base import Simulation


@ray.remote
class StripWorker:
    def __init__(
        self,
        worker_id: int,
        simulation: Simulation,
        strip_data: torch.Tensor,
        row_offset: int,
    ) -> None:
        self.worker_id = worker_id
        self.simulation = simulation
        self.row_offset = row_offset
        self.halo_size = simulation.halo_size
        self.params = simulation.default_params()

        # Device detection
        if torch.cuda.is_available():
            gpu_id = ray.get_gpu_ids()
            if gpu_id:
                self.device = torch.device(f"cuda:{gpu_id[0]}")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        # strip_data: (channels, owned_height, width) - no halos yet
        self.owned_height = strip_data.shape[1]
        self.width = strip_data.shape[2]

        # Allocate grid with halo space: (channels, owned_height + 2*halo, width)
        h = self.halo_size
        c = strip_data.shape[0]
        self.grid = torch.zeros(c, self.owned_height + 2 * h, self.width, device=self.device)
        # Place owned data in the interior
        self.grid[:, h : h + self.owned_height, :] = strip_data.to(self.device)

        # Precompute palette on device
        self.palette_lut = simulation.palette().to(self.device)

        self._last_step_ms: float = 0.0

    def step(self) -> None:
        """Run one simulation step on the owned strip."""
        t0 = time.perf_counter()
        h = self.halo_size

        # Horizontal circular padding for toroidal left-right wrapping
        padded = F.pad(self.grid, (h, h, 0, 0), mode="circular")

        # step() expects (C, H+2h, W+2h), returns (C, owned_H, W)
        result = self.simulation.step(padded, self.params)

        # Write result back into the interior of self.grid
        self.grid[:, h : h + self.owned_height, :] = result

        self._last_step_ms = (time.perf_counter() - t0) * 1000

    def get_top_boundary(self) -> torch.Tensor:
        """Return top h owned rows for the neighbor above."""
        h = self.halo_size
        return self.grid[:, h : h + h, :].cpu()

    def get_bottom_boundary(self) -> torch.Tensor:
        """Return bottom h owned rows for the neighbor below."""
        h = self.halo_size
        return self.grid[:, self.owned_height : self.owned_height + h, :].cpu()

    def set_top_halo(self, data: torch.Tensor) -> None:
        """Receive halo data from the neighbor above."""
        h = self.halo_size
        self.grid[:, :h, :] = data.to(self.device)

    def set_bottom_halo(self, data: torch.Tensor) -> None:
        """Receive halo data from the neighbor below."""
        h = self.halo_size
        self.grid[:, h + self.owned_height :, :] = data.to(self.device)

    def render(self) -> bytes:
        """Apply colormap, return raw RGB bytes of owned region (no halos)."""
        h = self.halo_size
        owned = self.grid[:, h : h + self.owned_height, :]
        vis = self.simulation.render_transform(owned)  # (H, W) in [0, 1]
        indices = (vis * 255).clamp(0, 255).byte()
        rgb = self.palette_lut[indices.long()]  # (H, W, 3) uint8
        return rgb.cpu().numpy().tobytes()

    def get_strip_data(self) -> torch.Tensor:
        """Return owned grid data (no halos) for repartitioning."""
        h = self.halo_size
        return self.grid[:, h : h + self.owned_height, :].cpu()

    def set_strip_data(self, data: torch.Tensor, row_offset: int) -> None:
        """Replace grid with new data after repartitioning."""
        h = self.halo_size
        self.owned_height = data.shape[1]
        self.row_offset = row_offset
        c = data.shape[0]
        self.grid = torch.zeros(c, self.owned_height + 2 * h, self.width, device=self.device)
        self.grid[:, h : h + self.owned_height, :] = data.to(self.device)

    def update_params(self, params: dict[str, float]) -> None:
        self.params.update(params)

    def perturb(self, row: int, col: int, channel: int, value: float, radius: int) -> None:
        """Write a value at grid coordinates (relative to this strip's owned region)."""
        h = self.halo_size
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc <= radius * radius:
                    r = row + dr + h
                    c = col + dc
                    if 0 <= r < self.grid.shape[1] and 0 <= c < self.width:
                        self.grid[channel, r, c] = value

    def get_metrics(self) -> dict[str, float]:
        return {
            "worker_id": float(self.worker_id),
            "step_ms": self._last_step_ms,
            "owned_height": float(self.owned_height),
            "device": 0.0,  # placeholder, will be string in metrics dict
        }

    def ping(self) -> bool:
        """Health check."""
        return True
