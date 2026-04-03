from __future__ import annotations

from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from gridlife.engine.coordinator import Coordinator
    from gridlife.simulations.base import Simulation

app = typer.Typer(name="gridlife", help="Distributed grid simulation engine")


@app.command()
def serve(
    sim: str = typer.Option("gray_scott", help="Simulation name or path to .py file"),
    width: int = typer.Option(512, help="Grid width"),
    height: int = typer.Option(512, help="Grid height"),
    workers: int = typer.Option(4, help="Number of workers"),
    host: str = typer.Option("0.0.0.0", help="Web server host"),
    port: int = typer.Option(8420, help="Web server port"),
    render_fps: int = typer.Option(20, help="Target render FPS"),
    max_speed: bool = typer.Option(False, help="No throttling, run as fast as possible"),
    steps_per_run: int = typer.Option(1000, help="Steps per play cycle (0 for unlimited)"),
    ray_address: str | None = typer.Option(None, help="Ray cluster address for distributed mode"),
    local_ray: bool = typer.Option(False, help="Use Ray actors locally (for testing Ray behavior)"),
) -> None:
    """Start the simulation engine with web UI."""
    import signal

    import uvicorn

    from gridlife.web.server import SimulationServer, create_app

    coordinator_factory = None
    use_ray = ray_address is not None or local_ray

    if use_ray:
        import os

        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

        import ray

        from gridlife.engine.ray_pool import RayWorkerPool

        if ray_address:
            ray.init(address=ray_address, ignore_reinit_error=True)
            typer.echo(f"Connected to Ray cluster: {ray.cluster_resources()}")
        else:
            available_cpus = os.cpu_count() or 4
            max_ray_cpus = max(2, available_cpus - 2)
            ray.init(num_cpus=max_ray_cpus, ignore_reinit_error=True)
            typer.echo(f"Local Ray mode: {max_ray_cpus} CPUs (reserving 2 for system)")

        def make_coordinator(simulation):  # type: ignore[no-untyped-def]
            from gridlife.engine.coordinator import Coordinator as Coord

            pool = RayWorkerPool(simulation, width, height, workers)
            return Coord(simulation, width, height, workers, pool=pool)

        coordinator_factory = make_coordinator
    else:
        typer.echo(f"Local mode: {workers} in-process workers")

    server = SimulationServer(
        sim_name=sim,
        width=width,
        height=height,
        num_workers=workers,
        render_fps=render_fps,
        throttle=not max_speed,
        steps_per_run=steps_per_run,
        coordinator_factory=coordinator_factory,
    )

    fastapi_app = create_app(server)

    def shutdown(sig: int, frame: object) -> None:
        import os

        typer.echo("\nShutting down...")
        server.running = False
        if server.coordinator:
            server.coordinator.shutdown()
        if use_ray:
            import ray

            ray.shutdown()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    typer.echo(f"Starting gridlife at http://{host}:{port}")
    typer.echo(f"Simulation: {sim} ({width}x{height}), {workers} workers")
    if steps_per_run > 0:
        typer.echo(f"Steps per play cycle: {steps_per_run}")
    uvicorn.run(fastapi_app, host=host, port=port, log_level="warning")


@app.command()
def run(
    sim: str = typer.Option("gray_scott", help="Simulation name or path to .py file"),
    width: int = typer.Option(512, help="Grid width"),
    height: int = typer.Option(512, help="Grid height"),
    workers: int = typer.Option(4, help="Number of workers"),
    steps: int = typer.Option(500, help="Number of steps to run"),
    output: str | None = typer.Option(None, help="Output file (.png, .gif)"),
    fps: int = typer.Option(20, help="GIF frame rate"),
    params: str | None = typer.Option(None, help="Parameter overrides as JSON string"),
    preset: str | None = typer.Option(None, help="Use a named preset"),
) -> None:
    """Run simulation headless and optionally produce output."""
    import json
    import time

    from gridlife.engine.coordinator import Coordinator as Coord

    sim_cls = _get_simulation(sim)
    if sim_cls is None:
        typer.echo(f"Unknown simulation: {sim}")
        raise typer.Exit(1)

    simulation = sim_cls()

    # Apply preset first, then parameter overrides
    active_params = simulation.default_params()
    if preset:
        if preset in simulation.presets:
            active_params.update(simulation.presets[preset])
            typer.echo(f"Using preset: {preset}")
        else:
            typer.echo(
                f"Unknown preset: {preset}. Available: {', '.join(simulation.presets.keys())}"
            )
            raise typer.Exit(1)

    if params:
        try:
            overrides = json.loads(params)
            active_params.update(overrides)
        except json.JSONDecodeError as err:
            typer.echo("Invalid JSON for --params")
            raise typer.Exit(1) from err

    coord = Coord(simulation, width, height, workers)
    coord.update_params(active_params)

    typer.echo(f"Running {simulation.name} ({width}x{height}), {workers} workers, {steps} steps")

    # Determine if we need to capture frames for GIF
    capturing = output is not None and output.endswith(".gif")
    frame_interval = 1
    frames: list[bytes] = []

    if capturing:
        total_frames = min(steps, fps * 30)  # cap at 30 seconds of GIF
        frame_interval = max(1, steps // total_frames)
        typer.echo(f"Capturing frame every {frame_interval} steps ({total_frames} frames)")

    t0 = time.perf_counter()

    for step in range(steps):
        coord.do_step()

        if capturing and step % frame_interval == 0:
            frame_data = _render_full_frame(coord, simulation, width, height)
            frames.append(frame_data)

        if steps >= 100 and step % (steps // 10) == 0 and step > 0:
            elapsed = time.perf_counter() - t0
            rate = step / elapsed
            typer.echo(f"  step {step}/{steps} ({rate:.0f} steps/s)")

    elapsed = time.perf_counter() - t0
    typer.echo(f"Done: {steps} steps in {elapsed:.2f}s ({steps / elapsed:.0f} steps/s)")

    if output is not None:
        if output.endswith(".png"):
            frame_data = _render_full_frame(coord, simulation, width, height)
            _save_png(frame_data, width, height, output)
            typer.echo(f"Saved PNG: {output}")

        elif output.endswith(".gif"):
            frame_data = _render_full_frame(coord, simulation, width, height)
            frames.append(frame_data)
            _save_gif(frames, width, height, output, fps)
            typer.echo(f"Saved GIF: {output} ({len(frames)} frames)")

        else:
            typer.echo(f"Unsupported output format: {output}. Use .png or .gif")

    coord.shutdown()


@app.command(name="list")
def list_sims() -> None:
    """List available simulations."""
    from gridlife.simulations.game_of_life import GameOfLife
    from gridlife.simulations.gray_scott import GrayScott
    from gridlife.simulations.lenia import Lenia
    from gridlife.simulations.smoothlife import SmoothLife

    for sim_cls in [GameOfLife, GrayScott, Lenia, SmoothLife]:
        s = sim_cls()
        params_str = ", ".join(s.params.keys()) if s.params else "none"
        typer.echo(f"  {s.name:20s} {s.description}")
        typer.echo(
            f"  {'':20s} Channels: {s.channels} | Halo: {s.halo_size} | Params: {params_str}"
        )
        if s.presets:
            typer.echo(f"  {'':20s} Presets: {', '.join(s.presets.keys())}")
        typer.echo()


def _get_simulation(name: str) -> type | None:
    from gridlife.simulations.game_of_life import GameOfLife
    from gridlife.simulations.gray_scott import GrayScott
    from gridlife.simulations.lenia import Lenia
    from gridlife.simulations.smoothlife import SmoothLife

    sims: dict[str, type] = {
        "game_of_life": GameOfLife,
        "gray_scott": GrayScott,
        "lenia": Lenia,
        "smoothlife": SmoothLife,
    }
    return sims.get(name)


def _render_full_frame(
    coord: Coordinator,
    simulation: Simulation,
    width: int,
    height: int,
) -> bytes:
    """Render the full grid as RGB bytes."""
    import numpy as np

    strips = coord.collect_frame()
    raw = b"".join(strips)
    indices = np.frombuffer(raw, dtype=np.uint8)
    palette = simulation.palette().numpy()
    rgb = palette[indices].reshape(height, width, 3)
    return rgb.tobytes()


def _save_png(rgb_bytes: bytes, width: int, height: int, path: str) -> None:
    from PIL import Image

    img = Image.frombytes("RGB", (width, height), rgb_bytes)
    img.save(path, format="PNG")


def _save_gif(
    frames_rgb: list[bytes],
    width: int,
    height: int,
    path: str,
    fps: int,
) -> None:
    from PIL import Image

    images = []
    for frame_bytes in frames_rgb:
        img = Image.frombytes("RGB", (width, height), frame_bytes)
        images.append(img)

    if not images:
        return

    duration_ms = 1000 // fps
    images[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
