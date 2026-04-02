import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import ray
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from gridlife.engine.coordinator import Coordinator
from gridlife.simulations.base import Simulation
from gridlife.simulations.game_of_life import GameOfLife
from gridlife.simulations.gray_scott import GrayScott
from gridlife.viz.encoder import encode_frame_jpeg

logger = logging.getLogger("gridlife.web")

STATIC_DIR = Path(__file__).parent / "static"

SIMULATIONS: dict[str, type[Simulation]] = {
    "game_of_life": GameOfLife,
    "gray_scott": GrayScott,
}


class SimulationServer:
    """Manages the simulation lifecycle and WebSocket streaming."""

    def __init__(
        self,
        sim_name: str,
        width: int,
        height: int,
        num_workers: int,
        gpu: bool,
        render_fps: int,
        render_resolution: int,
    ) -> None:
        self.width = width
        self.height = height
        self.num_workers = num_workers
        self.gpu = gpu
        self.render_fps = render_fps
        self.render_resolution = render_resolution

        self.coordinator: Coordinator | None = None
        self.simulation: Simulation | None = None
        self.running = False
        self.clients: list[WebSocket] = []

        self._init_simulation(sim_name)

    def _init_simulation(self, sim_name: str) -> None:
        if self.coordinator is not None:
            self.coordinator.shutdown()

        sim_cls = SIMULATIONS.get(sim_name)
        if sim_cls is None:
            raise ValueError(f"Unknown simulation: {sim_name}")

        self.simulation = sim_cls()
        self.coordinator = Coordinator(
            self.simulation,
            self.width,
            self.height,
            self.num_workers,
            gpu=self.gpu,
        )

    def collect_frame(self) -> bytes:
        """Collect rendered strips from workers and encode as JPEG."""
        if self.coordinator is None or self.simulation is None:
            return b""

        strips_bytes = ray.get([w.render.remote() for w in self.coordinator.workers])

        row_ranges = self.coordinator.row_ranges
        strip_heights = [end - start for start, end in row_ranges]

        full_rgb = b""
        for strip_b in strips_bytes:
            full_rgb += strip_b

        total_height = sum(strip_heights)
        return encode_frame_jpeg(full_rgb, self.width, total_height)

    async def step_loop(self) -> None:
        """Main simulation + render loop."""
        render_interval = max(1, 100 // self.render_fps)
        step_in_batch = 0

        while self.running:
            if self.coordinator is None:
                await asyncio.sleep(0.1)
                continue

            self.coordinator.do_step()
            step_in_batch += 1

            if step_in_batch >= render_interval:
                step_in_batch = 0
                try:
                    frame = self.collect_frame()
                    await self._broadcast_binary(frame)
                    metrics = self.coordinator.get_metrics()
                    await self._broadcast_json({"type": "metrics", "data": metrics})
                except Exception as e:
                    logger.error(f"Frame collection error: {e}")

            await asyncio.sleep(0)

    async def _broadcast_binary(self, data: bytes) -> None:
        disconnected = []
        for ws in self.clients:
            try:
                await ws.send_bytes(data)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.clients.remove(ws)

    async def _broadcast_json(self, data: dict[str, Any]) -> None:
        disconnected = []
        for ws in self.clients:
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.clients.remove(ws)

    async def handle_message(self, msg: dict[str, Any]) -> None:
        msg_type = msg.get("type")

        if msg_type == "play":
            if not self.running:
                self.running = True
                asyncio.create_task(self.step_loop())

        elif msg_type == "pause":
            self.running = False

        elif msg_type == "set_params" and self.coordinator is not None:
            params = msg.get("params", {})
            self.coordinator.update_params(params)

        elif msg_type == "set_workers" and self.coordinator is not None:
            count = msg.get("count", self.num_workers)
            self.running = False
            await asyncio.sleep(0.05)
            self.coordinator.repartition(count)
            self.num_workers = count

        elif msg_type == "switch_sim":
            sim_name = msg.get("name", "game_of_life")
            self.running = False
            await asyncio.sleep(0.05)
            self._init_simulation(sim_name)
            await self._broadcast_json(
                {
                    "type": "sim_info",
                    "data": self.sim_info(),
                }
            )

        elif msg_type == "reset":
            self.running = False
            await asyncio.sleep(0.05)
            if self.simulation is not None:
                sim_name = self.simulation.name
                self._init_simulation(sim_name)

        elif msg_type == "perturb" and self.coordinator is not None:
            row = msg.get("row", 0)
            col = msg.get("col", 0)
            channel = msg.get("channel", 0)
            value = msg.get("value", 1.0)
            radius = msg.get("radius", 3)
            for i, (start, end) in enumerate(self.coordinator.row_ranges):
                if start <= row < end:
                    local_row = row - start
                    ray.get(
                        self.coordinator.workers[i].perturb.remote(
                            local_row, col, channel, value, radius
                        )
                    )
                    break

    def sim_info(self) -> dict[str, Any]:
        if self.simulation is None:
            return {}
        return {
            "name": self.simulation.name,
            "description": self.simulation.description,
            "channels": self.simulation.channels,
            "params": {
                k: {
                    "default": v.default,
                    "min": v.min,
                    "max": v.max,
                    "step": v.step,
                    "description": v.description,
                }
                for k, v in self.simulation.params.items()
            },
            "presets": self.simulation.presets,
            "pixelated": self.simulation.pixelated,
        }

    def get_simulations_list(self) -> list[dict[str, Any]]:
        result = []
        for _name, cls in SIMULATIONS.items():
            sim = cls()
            result.append(
                {
                    "name": sim.name,
                    "description": sim.description,
                    "channels": sim.channels,
                    "halo_size": sim.halo_size,
                    "params": {
                        k: {"default": v.default, "min": v.min, "max": v.max, "step": v.step}
                        for k, v in sim.params.items()
                    },
                    "presets": sim.presets,
                }
            )
        return result


def create_app(server: SimulationServer) -> FastAPI:
    app = FastAPI(title="gridlife")

    @app.get("/")
    async def index() -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/simulations")
    async def list_simulations() -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
        return server.get_simulations_list()

    @app.get("/api/status")
    async def status() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        metrics = server.coordinator.get_metrics() if server.coordinator else {}
        return {
            "running": server.running,
            "simulation": server.sim_info(),
            "metrics": metrics,
        }

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:  # pyright: ignore[reportUnusedFunction]
        await ws.accept()
        server.clients.append(ws)

        await ws.send_text(
            json.dumps(
                {
                    "type": "sim_info",
                    "data": server.sim_info(),
                }
            )
        )

        try:
            while True:
                text = await ws.receive_text()
                try:
                    msg = json.loads(text)
                    await server.handle_message(msg)
                except json.JSONDecodeError:
                    pass
        except WebSocketDisconnect:
            if ws in server.clients:
                server.clients.remove(ws)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app
