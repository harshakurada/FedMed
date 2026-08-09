from __future__ import annotations

from server.dashboard.events import DashboardEvent, EventType
from server.dashboard.mock_events import mock_mode_enabled, run_mock_event_sequence
from server.dashboard.state import DashboardState
from server.dashboard.websocket_server import DashboardWebSocketServer


def test_mock_mode_defaults_to_disabled(monkeypatch) -> None:
    monkeypatch.delenv("DASHBOARD_MOCK_MODE", raising=False)
    assert mock_mode_enabled() is False


def test_mock_mode_explicit_enable(monkeypatch) -> None:
    monkeypatch.setenv("DASHBOARD_MOCK_MODE", "true")
    assert mock_mode_enabled() is True
    monkeypatch.setenv("DASHBOARD_MOCK_MODE", "false")
    assert mock_mode_enabled() is False


def test_mock_event_sequence_is_self_contained_and_produces_valid_events() -> None:
    state = DashboardState()
    server = DashboardWebSocketServer(state, host="127.0.0.1", port=0)  # never started -- emit() just updates state

    run_mock_event_sequence(server)

    assert state.system_status in ("Idle", "Training")  # ended after round 2 completed
    assert len(state.hospitals) == 3
    assert state.current_round == 2
    assert isinstance(state.global_dice, float)
