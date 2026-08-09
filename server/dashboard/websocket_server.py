"""The dashboard WebSocket server -- Python's `websockets` library (already an approved,
declared dependency since Module 1; not Socket.IO). Sends the current snapshot to every
newly-connected dashboard immediately, then broadcasts every `DashboardEvent` as it's
emitted. No training/model/encryption/privacy/aggregation logic lives here -- this only
transports already-validated, already-safe events (see `events.py`).

Explicit `start()`/`stop()` only -- importing this module never starts a server.
"""

from __future__ import annotations

import asyncio
import json
import logging

import websockets
from websockets.asyncio.server import Server, ServerConnection

from server.dashboard.events import DashboardEvent
from server.dashboard.state import DashboardState

logger = logging.getLogger("fedmed.dashboard.websocket")


class DashboardWebSocketServer:
    """Implements `EventSink` (`.emit`) so `server/federated/experiment.py`'s round loop
    can use it directly -- the round loop never imports `websockets` itself."""

    def __init__(self, state: DashboardState, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.state = state
        self.host = host
        self.port = port
        self._connections: set[ServerConnection] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Server | None = None

    async def _handler(self, connection: ServerConnection) -> None:
        self._connections.add(connection)
        logger.info("dashboard connected (%d total)", len(self._connections))
        try:
            await connection.send(json.dumps({"type": "snapshot", "data": self.state.snapshot()}))
            async for _message in connection:
                pass  # the dashboard is read-only from the server's perspective
        except websockets.ConnectionClosed:
            pass
        finally:
            self._connections.discard(connection)
            logger.info("dashboard disconnected (%d remaining)", len(self._connections))

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._server = await websockets.serve(self._handler, self.host, self.port)
        logger.info("dashboard WebSocket server listening on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _broadcast_async(self, event: DashboardEvent) -> None:
        self.state.apply(event)
        if not self._connections:
            return
        message = event.to_json()
        results = await asyncio.gather(
            *(connection.send(message) for connection in list(self._connections)), return_exceptions=True
        )
        for connection, result in zip(list(self._connections), results):
            if isinstance(result, Exception):
                self._connections.discard(connection)

    def emit(self, event: DashboardEvent) -> None:
        """Safe to call from any thread (the round loop typically runs in a worker
        thread via `asyncio.to_thread` while this server's loop runs on the main
        thread -- see `run_dashboard_backend.py`)."""
        if self._loop is None:
            self.state.apply(event)  # not started yet -- still reflected in a later snapshot
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_async(event), self._loop)
