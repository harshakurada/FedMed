"""Real `websockets` server tests -- an actual server on an ephemeral localhost port,
an actual client connection, no mocking of the transport (same pattern Module 8 used
for its real mTLS gRPC tests)."""

from __future__ import annotations

import asyncio
import json
import socket

import pytest
import websockets

from server.dashboard.events import DashboardEvent, EventType
from server.dashboard.state import DashboardState
from server.dashboard.websocket_server import DashboardWebSocketServer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_client_receives_snapshot_immediately_on_connect() -> None:
    async def scenario() -> None:
        state = DashboardState()
        state.system_status = "Ready"
        server = DashboardWebSocketServer(state, host="127.0.0.1", port=_free_port())
        await server.start()
        try:
            async with websockets.connect(f"ws://{server.host}:{server.port}") as client:
                message = json.loads(await asyncio.wait_for(client.recv(), timeout=5.0))
                assert message["type"] == "snapshot"
                assert message["data"]["system_status"] == "Ready"
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_broadcast_event_reaches_connected_client() -> None:
    async def scenario() -> None:
        state = DashboardState()
        server = DashboardWebSocketServer(state, host="127.0.0.1", port=_free_port())
        await server.start()
        try:
            async with websockets.connect(f"ws://{server.host}:{server.port}") as client:
                await client.recv()  # discard snapshot
                server.emit(DashboardEvent(event_type=EventType.ROUND_STARTED, source="server", round=1))
                message = json.loads(await asyncio.wait_for(client.recv(), timeout=5.0))
                assert message["type"] == "event"
                assert message["data"]["event_type"] == "ROUND_STARTED"
                assert message["data"]["round"] == 1
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_multiple_connected_clients_all_receive_the_broadcast() -> None:
    async def scenario() -> None:
        state = DashboardState()
        server = DashboardWebSocketServer(state, host="127.0.0.1", port=_free_port())
        await server.start()
        try:
            async with websockets.connect(f"ws://{server.host}:{server.port}") as client_a, \
                    websockets.connect(f"ws://{server.host}:{server.port}") as client_b:
                await client_a.recv()
                await client_b.recv()
                server.emit(DashboardEvent(event_type=EventType.CLIENT_CONNECTED, source="hospital_a"))
                msg_a = json.loads(await asyncio.wait_for(client_a.recv(), timeout=5.0))
                msg_b = json.loads(await asyncio.wait_for(client_b.recv(), timeout=5.0))
                assert msg_a["data"]["event_type"] == "CLIENT_CONNECTED"
                assert msg_b["data"]["event_type"] == "CLIENT_CONNECTED"
        finally:
            await server.stop()

    asyncio.run(scenario())


def test_emit_before_start_updates_state_without_raising() -> None:
    state = DashboardState()
    server = DashboardWebSocketServer(state, host="127.0.0.1", port=_free_port())
    server.emit(DashboardEvent(event_type=EventType.SYSTEM_READY, source="server"))
    assert state.system_status == "Ready"


def test_state_apply_rejects_malformed_event_gracefully() -> None:
    # A malformed/invalid event never even constructs (events.py's own validation) --
    # verified here at the boundary this module actually owns.
    from server.dashboard.events import DashboardEventError

    with pytest.raises(DashboardEventError):
        DashboardEvent(event_type="BOGUS", source="server")
