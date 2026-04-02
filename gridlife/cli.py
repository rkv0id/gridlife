import typer

app = typer.Typer(name="gridlife", help="Distributed grid simulation engine")


@app.command()
def serve(
    sim: str = typer.Option("gray_scott", help="Simulation name or path to .py file"),
    width: int = typer.Option(1024, help="Grid width"),
    height: int = typer.Option(1024, help="Grid height"),
    workers: int | None = typer.Option(None, help="Number of workers (auto-detect if omitted)"),
    host: str = typer.Option("0.0.0.0", help="Web server host"),
    port: int = typer.Option(8420, help="Web server port"),
) -> None:
    """Start the simulation engine with web UI."""
    typer.echo(f"Starting gridlife with {sim} ({width}x{height})")
    typer.echo("Web server not implemented yet.")


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
