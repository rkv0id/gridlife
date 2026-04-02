import typer

app = typer.Typer(name="gridlife", help="Distributed grid simulation engine")


@app.command()
def serve(
    sim: str = typer.Option("gray_scott", help="Simulation name or path to .py file"),
    width: int = typer.Option(512, help="Grid width"),
    height: int = typer.Option(512, help="Grid height"),
    workers: int = typer.Option(2, help="Number of workers"),
    host: str = typer.Option("0.0.0.0", help="Web server host"),
    port: int = typer.Option(8420, help="Web server port"),
    render_fps: int = typer.Option(20, help="Target render FPS"),
    render_res: int = typer.Option(1024, help="Max render resolution"),
    no_gpu: bool = typer.Option(False, help="Force CPU-only workers"),
) -> None:
    """Start the simulation engine with web UI."""
    import os

    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

    import ray
    import uvicorn

    from gridlife.web.server import SimulationServer, create_app

    ray.init(
        ignore_reinit_error=True,
        runtime_env={"excludes": ["**"]},
    )
    typer.echo(f"Ray initialized: {ray.cluster_resources()}")

    server = SimulationServer(
        sim_name=sim,
        width=width,
        height=height,
        num_workers=workers,
        gpu=not no_gpu,
        render_fps=render_fps,
        render_resolution=render_res,
    )

    fastapi_app = create_app(server)
    typer.echo(f"Starting gridlife at http://{host}:{port}")
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
