import typer

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
            from gridlife.engine.coordinator import Coordinator

            pool = RayWorkerPool(simulation, width, height, workers)
            return Coordinator(simulation, width, height, workers, pool=pool)

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
        typer.echo("\nShutting down...")
        server.running = False
        if server.coordinator:
            server.coordinator.shutdown()
        if use_ray:
            import ray

            ray.shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    typer.echo(f"Starting gridlife at http://{host}:{port}")
    typer.echo(f"Simulation: {sim} ({width}x{height}), {workers} workers")
    if steps_per_run > 0:
        typer.echo(f"Steps per play cycle: {steps_per_run}")
    uvicorn.run(fastapi_app, host=host, port=port, log_level="warning")


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
