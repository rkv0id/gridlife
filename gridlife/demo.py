"""
Single-process simulation runner for local testing.
No Ray, no distribution - runs the sim on a single grid.

    python -m gridlife.demo --sim game_of_life --steps 100 --size 64
    python -m gridlife.demo --sim gray_scott --steps 500 --size 128 --save frame.png
"""

import argparse
import io
import time

import torch
import torch.nn.functional as F

from gridlife.simulations.base import Simulation
from gridlife.simulations.game_of_life import GameOfLife
from gridlife.simulations.gray_scott import GrayScott

SIMS: dict[str, type[Simulation]] = {
    "game_of_life": GameOfLife,
    "gray_scott": GrayScott,
}


def run_single(
    sim: Simulation,
    height: int,
    width: int,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Run simulation for N steps on a single grid. Returns final grid state."""
    grid = sim.init_grid(height, width, device)
    params = sim.default_params()
    h = sim.halo_size

    for _ in range(steps):
        padded = F.pad(grid, (h, h, h, h), mode="circular")
        grid = sim.step(padded, params)

    return grid


def render_frame(sim: Simulation, grid: torch.Tensor) -> bytes:
    """Render grid to RGB PNG bytes."""
    from PIL import Image

    vis = sim.render_transform(grid)
    indices = (vis * 255).clamp(0, 255).byte().cpu()
    pal = sim.palette()
    rgb = pal[indices.long()]

    img = Image.fromarray(rgb.numpy(), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="gridlife single-process demo")
    parser.add_argument("--sim", choices=list(SIMS.keys()), default="game_of_life")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--size", type=int, default=64, help="Grid width and height")
    parser.add_argument("--save", type=str, default=None, help="Save final frame to PNG")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    sim = SIMS[args.sim]()
    device = torch.device(args.device)

    print(f"Running {sim.name} on {device}, {args.size}x{args.size}, {args.steps} steps")

    t0 = time.perf_counter()
    grid = run_single(sim, args.size, args.size, args.steps, device)
    elapsed = time.perf_counter() - t0

    print(f"Done in {elapsed:.3f}s ({args.steps / elapsed:.0f} steps/s)")
    print(f"Grid shape: {grid.shape}, range: [{grid.min():.4f}, {grid.max():.4f}]")

    if args.save:
        frame_bytes = render_frame(sim, grid)
        with open(args.save, "wb") as f:
            f.write(frame_bytes)
        print(f"Saved frame to {args.save} ({len(frame_bytes)} bytes)")


if __name__ == "__main__":
    main()
