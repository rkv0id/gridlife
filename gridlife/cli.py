import typer

app = typer.Typer(name="gridlife", help="Distributed grid simulation engine")


@app.command()
def serve(
    sim: str = typer.Option("gray_scott", help="Simulation name or path to .py file"),
    width: int = typer.Option(256, help="Grid width"),
    height: int = typer.Option(256, help="Grid height"),
    workers: int = typer.Option(2, help="Number of workers"),
    host: str = typer.Option("0.0.0.0", help="Web server host"),
    port: int = typer.Option(8420, help="Web server port"),
    render_fps: int = typer.Option(20, help="Target render FPS"),
    render_res: int = typer.Option(1024, help="Max render resolution"),
    gpu: bool = typer.Option(False, help="Request GPU workers (CUDA only)"),
    max_speed: bool = typer.Option(False, help="No throttling, run as fast as possible"),
) -> None:
    """Start the simulation engine with web UI."""
    import os
    import signal

    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

    import ray
    import uvicorn

    from gridlife.web.server import SimulationServer, create_app

    # Reserve 2 cores for the OS and this process so the system stays responsive
    available_cpus = os.cpu_count() or 4
    max_ray_cpus = max(2, available_cpus - 2)
    ray.init(num_cpus=max_ray_cpus, ignore_reinit_error=True)
    typer.echo(f"Ray initialized with {max_ray_cpus} CPUs (reserving 2 for system)")

    server = SimulationServer(
        sim_name=sim,
        width=width,
        height=height,
        num_workers=workers,
        gpu=gpu,
        render_fps=render_fps,
        render_resolution=render_res,
        throttle=not max_speed,
    )

    fastapi_app = create_app(server)

    # Graceful shutdown: stop the sim loop, kill workers, then exit.
    # Without this, Ray child processes survive Ctrl+C.
    def shutdown(sig: int, frame: object) -> None:
        typer.echo("\nShutting down...")
        server.running = False
        if server.coordinator:
            server.coordinator.shutdown()
        ray.shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    typer.echo(f"Starting gridlife at http://{host}:{port}")
    typer.echo(f"Simulation: {sim} ({width}x{height}), {workers} workers")
    if max_speed:
        typer.echo("Throttling disabled - running at max speed")
    uvicorn.run(fastapi_app, host=host, port=port, log_level="warning")


@app.command(name="list")
def list_sims() -> None:
    """List available simulations."""
    from gridlife.simulations.game_of_life import GameOfLife
    from gridlife.simulations.gray_scott import GrayScott

    for sim_cls in [GameOfLife, GrayScott]:
        s = sim_cls()
        params_str = ", ".join(s.params.keys()) if s.params else "none"
        typer.echo(f"  {s.name:20s} {s.description}")
        typer.echo(
            f"  {'':20s} Channels: {s.channels} | Halo: {s.halo_size} | Params: {params_str}"
        )
        if s.presets:
            typer.echo(f"  {'':20s} Presets: {', '.join(s.presets.keys())}")
        typer.echo()
