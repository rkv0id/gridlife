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
from gridlife.viz.encoder import encode_frame_message, encode_palette_message

logger = logging.getLogger("gridlife.web")

STATIC_DIR = Path(__file__).parent / "static"

SIMULATIONS: dict[str, type[Simulation]] = {
    "game_of_life": GameOfLife,
    "gray_scott": GrayScott,
}


class SimulationServer:
    """
    Bridges the browser UI and the distributed simulation engine.

    Owns the Coordinator, manages the step/render loop, handles
    WebSocket clients, and routes control messages (play, pause,
    parameter changes, repartitioning, perturbation).
    """

    def __init__(
        self,
        sim_name: str,
        width: int,
        height: int,
        num_workers: int,
        gpu: bool,
        render_fps: int,
        render_resolution: int,
        throttle: bool = True,
        steps_per_run: int = 500,
    ) -> None:
        self.width = width
        self.height = height
        self.num_workers = num_workers
        self.gpu = gpu
        self.render_fps = render_fps
        self.render_resolution = render_resolution
        self.throttle = throttle
        self.steps_per_run = steps_per_run

        self.coordinator: Coordinator | None = None
        self.simulation: Simulation | None = None
        self.running = False
        self.clients: list[WebSocket] = []

        self._init_simulation(sim_name)

    def _init_simulation(self, sim_name: str) -> None:
        """Tear down any existing simulation and bootstrap a new one."""
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

    def _get_palette_bytes(self) -> bytes:
        """Get the current simulation's palette as raw bytes (256 * 3 = 768 bytes)."""
        if self.simulation is None:
            return b"\x00" * 768
        return self.simulation.palette().numpy().tobytes()

    def collect_frame(self) -> bytes:
        """
        Gather rendered uint8 strips from all workers, concatenate,
        and wrap in a binary frame message.
        """
        if self.coordinator is None or self.simulation is None:
            return b""

        strips_bytes = ray.get([w.render.remote() for w in self.coordinator.workers])

        raw = b""
        for strip_b in strips_bytes:
            raw += strip_b

        return encode_frame_message(raw, self.width, self.height)

    async def step_loop(self) -> None:
        """
        Runs simulation steps in a background thread to avoid blocking
        the asyncio event loop. Stops after steps_per_run steps (if set)
        or when paused. User clicks Play again to run another batch.
        """
        import concurrent.futures

        loop = asyncio.get_event_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        target_frame_time = 1.0 / self.render_fps
        steps_per_frame = max(1, 100 // self.render_fps)
        steps_done = 0
        unlimited = self.steps_per_run == 0

        def run_batch() -> int:
            """Blocking work that runs in the thread pool."""
            if self.coordinator is None:
                return 0
            count = 0
            for _ in range(steps_per_frame):
                if not self.running:
                    break
                self.coordinator.do_step()
                count += 1
            return count

        while self.running:
            if self.coordinator is None:
                await asyncio.sleep(0.1)
                continue

            frame_start = loop.time()

            batch_count = await loop.run_in_executor(executor, run_batch)
            steps_done += batch_count

            if not self.running:
                break

            # Check step limit
            if not unlimited and steps_done >= self.steps_per_run:
                self.running = False
                await self._broadcast_json(
                    {
                        "type": "status",
                        "data": {"paused": True, "reason": "step_limit"},
                    }
                )

            try:
                frame = await loop.run_in_executor(executor, self.collect_frame)
                await self._broadcast_binary(frame)
                metrics = self.coordinator.get_metrics()
                await self._broadcast_json({"type": "metrics", "data": metrics})
            except Exception as e:
                logger.error(f"Frame collection error: {e}")

            if self.throttle:
                elapsed = loop.time() - frame_start
                sleep_time = target_frame_time - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(0)

        executor.shutdown(wait=False)

    async def _broadcast_binary(self, data: bytes) -> None:
        """Send binary data to all WebSocket clients, removing any that have disconnected."""
        disconnected = []
        for ws in self.clients:
            try:
                await ws.send_bytes(data)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.clients.remove(ws)

    async def _broadcast_json(self, data: dict[str, Any]) -> None:
        """Send JSON data to all WebSocket clients, removing any that have disconnected."""
        disconnected = []
        for ws in self.clients:
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.clients.remove(ws)

    async def send_palette(self, ws: WebSocket | None = None) -> None:
        """Send palette to one client or broadcast to all."""
        msg = encode_palette_message(self._get_palette_bytes())
        if ws is not None:
            await ws.send_bytes(msg)
        else:
            await self._broadcast_binary(msg)

    async def _send_frame(self) -> None:
        """Render and broadcast a single frame. Used after reset, sim switch, repartition."""
        try:
            frame = self.collect_frame()
            await self._broadcast_binary(frame)
            if self.coordinator:
                metrics = self.coordinator.get_metrics()
                await self._broadcast_json({"type": "metrics", "data": metrics})
        except Exception as e:
            logger.error(f"Frame send error: {e}")

    async def handle_message(self, msg: dict[str, Any]) -> None:
        """
        Route an incoming WebSocket control message to the appropriate handler.
        Messages are JSON with a "type" field. Supported types:
        play, pause, set_params, set_workers, switch_sim, reset, perturb.
        """
        msg_type = msg.get("type")

        if msg_type == "play":
            if not self.running:
                self.running = True
                asyncio.create_task(self.step_loop())
                await self._broadcast_json({"type": "status", "data": {"paused": False}})

        elif msg_type == "pause":
            self.running = False
            await self._broadcast_json({"type": "status", "data": {"paused": True}})

        elif msg_type == "set_params" and self.coordinator is not None:
            params = msg.get("params", {})
            self.coordinator.update_params(params)

        elif msg_type == "set_workers" and self.coordinator is not None:
            count = msg.get("count", self.num_workers)
            self.running = False
            await asyncio.sleep(0.05)
            self.coordinator.repartition(count)
            self.num_workers = count
            await self._send_frame()

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
            await self.send_palette()
            await self._send_frame()

        elif msg_type == "reset":
            self.running = False
            await asyncio.sleep(0.05)
            if self.simulation is not None:
                sim_name = self.simulation.name
                self._init_simulation(sim_name)
            await self._send_frame()

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
        """Serialize current simulation metadata for the browser client."""
        if self.simulation is None:
            return {}
        return {
            "name": self.simulation.name,
            "description": self.simulation.description,
            "channels": self.simulation.channels,
            "width": self.width,
            "height": self.height,
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
        """Return metadata for all registered simulations (for the sim selector dropdown)."""
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
    """
    Build the FastAPI application. Registers HTTP routes for static files
    and API endpoints, plus the WebSocket endpoint for live streaming.
    """
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

        # Send simulation info, palette, and initial frame on connect
        await ws.send_text(
            json.dumps(
                {
                    "type": "sim_info",
                    "data": server.sim_info(),
                }
            )
        )
        await server.send_palette(ws)
        try:
            frame = server.collect_frame()
            await ws.send_bytes(frame)
        except Exception:
            pass

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
