from __future__ import annotations

import time
from typing import TYPE_CHECKING

import ray

from gridlife.simulations.base import Simulation

if TYPE_CHECKING:
    import torch


@ray.remote
class StripWorker:
    """
    Ray actor that owns a horizontal strip of the simulation grid.

    Each worker holds a buffer shaped (channels, owned_height + 2*halo, width).
    The interior region contains the owned rows. The top and bottom margins
    hold halo (ghost) data copied from neighboring workers before each step.

    Device selection is automatic: CUDA if Ray assigned a GPU, MPS on Apple
    Silicon, CPU otherwise. All tensor operations run on the selected device.
    """

    def __init__(
        self,
        worker_id: int,
        simulation: Simulation,
        strip_data: torch.Tensor,
        row_offset: int,
    ) -> None:
        import torch

        self.worker_id = worker_id
        self.simulation = simulation
        self.row_offset = row_offset
        self.halo_size = simulation.halo_size
        self.params = simulation.default_params()

        # CUDA > MPS > CPU. Ray only tracks CUDA GPUs as resources,
        # but MPS is usable for torch ops without Ray knowing about it.
        if torch.cuda.is_available():
            gpu_ids = ray.get_gpu_ids()
            if gpu_ids:
                self.device = torch.device(f"cuda:{gpu_ids[0]}")
            else:
                self.device = torch.device("cpu")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.owned_height = strip_data.shape[1]
        self.width = strip_data.shape[2]

        # Allocate buffer with halo margins on top and bottom
        h = self.halo_size
        c = strip_data.shape[0]
        self.grid = torch.zeros(c, self.owned_height + 2 * h, self.width, device=self.device)
        self.grid[:, h : h + self.owned_height, :] = strip_data.to(self.device)

        self._last_step_ms: float = 0.0
        self._device_name: str = str(self.device)

    def step(self) -> None:
        """
        Run one simulation step. Assumes halos are already populated
        by a prior exchange. Pads horizontally (circular/toroidal),
        calls the simulation's step function, and writes the result
        back into the owned interior of the buffer.
        """
        import torch.nn.functional as F

        t0 = time.perf_counter()
        h = self.halo_size

        padded = F.pad(self.grid, (h, h, 0, 0), mode="circular")
        result = self.simulation.step(padded, self.params)
        self.grid[:, h : h + self.owned_height, :] = result

        self._last_step_ms = (time.perf_counter() - t0) * 1000

    def get_top_boundary(self) -> torch.Tensor:
        """Return the top h owned rows (sent to neighbor above as their bottom halo)."""
        h = self.halo_size
        return self.grid[:, h : h + h, :].cpu()

    def get_bottom_boundary(self) -> torch.Tensor:
        """Return the bottom h owned rows (sent to neighbor below as their top halo)."""
        h = self.halo_size
        return self.grid[:, self.owned_height : self.owned_height + h, :].cpu()

    def set_top_halo(self, data: torch.Tensor) -> None:
        """Write halo data into the top margin (received from neighbor above)."""
        h = self.halo_size
        self.grid[:, :h, :] = data.to(self.device)

    def set_bottom_halo(self, data: torch.Tensor) -> None:
        """Write halo data into the bottom margin (received from neighbor below)."""
        h = self.halo_size
        self.grid[:, h + self.owned_height :, :] = data.to(self.device)

    def render(self) -> bytes:
        """
        Apply render_transform to get values in [0,1], quantize to uint8.
        Returns raw uint8 bytes (height * width), NOT RGB.
        The browser applies the palette client-side.
        """
        h = self.halo_size
        owned = self.grid[:, h : h + self.owned_height, :]
        vis = self.simulation.render_transform(owned)
        indices = (vis * 255).clamp(0, 255).byte()
        return indices.cpu().numpy().tobytes()

    def get_strip_data(self) -> torch.Tensor:
        """Return owned grid data without halos. Used for repartitioning and grid collection."""
        h = self.halo_size
        return self.grid[:, h : h + self.owned_height, :].cpu()

    def set_strip_data(self, data: torch.Tensor, row_offset: int) -> None:
        """Replace this worker's grid after repartitioning. Reallocates the buffer."""
        import torch

        h = self.halo_size
        self.owned_height = data.shape[1]
        self.row_offset = row_offset
        c = data.shape[0]
        self.grid = torch.zeros(c, self.owned_height + 2 * h, self.width, device=self.device)
        self.grid[:, h : h + self.owned_height, :] = data.to(self.device)

    def update_params(self, params: dict[str, float]) -> None:
        """Update simulation parameters from UI sliders. Takes effect on the next step."""
        self.params.update(params)

    def perturb(self, row: int, col: int, channel: int, value: float, radius: int) -> None:
        """
        Write a circular brush of values into the grid.
        Row is relative to this worker's owned region (not global grid coords).
        """
        h = self.halo_size
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc <= radius * radius:
                    r = row + dr + h
                    c = col + dc
                    if 0 <= r < self.grid.shape[1] and 0 <= c < self.width:
                        self.grid[channel, r, c] = value

    def get_metrics(self) -> dict[str, float | str]:
        return {
            "worker_id": float(self.worker_id),
            "step_ms": self._last_step_ms,
            "owned_height": float(self.owned_height),
            "device": self._device_name,
        }

    def ping(self) -> bool:
        """Health check. Returns True if the actor is alive and responsive."""
        return True
