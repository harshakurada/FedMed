from __future__ import annotations

from server.dashboard.events import DashboardEvent, EventType
from server.dashboard.state import DEFAULT_EVENT_HISTORY_LIMIT, DashboardState


def test_round_started_updates_system_and_round_status() -> None:
    state = DashboardState()
    state.apply(DashboardEvent(event_type=EventType.ROUND_STARTED, source="server", round=1, payload={"num_rounds": 3}))
    assert state.current_round == 1
    assert state.num_rounds == 3
    assert state.system_status == "Training"


def test_client_connected_then_training_then_completed_transitions() -> None:
    state = DashboardState()
    state.apply(DashboardEvent(event_type=EventType.CLIENT_CONNECTED, source="hospital_a"))
    assert state.hospitals["hospital_a"].connection_status == "Online"

    state.apply(DashboardEvent(event_type=EventType.CLIENT_TRAINING, source="hospital_a", round=1))
    assert state.hospitals["hospital_a"].connection_status == "Training"

    state.apply(
        DashboardEvent(
            event_type=EventType.CLIENT_TRAINING_COMPLETED, source="hospital_a", round=1,
            payload={"num_examples": 5, "train_loss": 0.4, "train_dice": 0.3, "train_iou": 0.2},
        )
    )
    hospital = state.hospitals["hospital_a"]
    assert hospital.connection_status == "Completed"
    assert hospital.train_dice == 0.3
    assert hospital.num_examples == 5


def test_client_failed_shows_error_status() -> None:
    state = DashboardState()
    state.apply(DashboardEvent(event_type=EventType.CLIENT_CONNECTED, source="hospital_b"))
    state.apply(DashboardEvent(event_type=EventType.CLIENT_FAILED, source="hospital_b", round=1, payload={"reason": "fit_error"}))
    assert state.hospitals["hospital_b"].connection_status == "Error"


def test_client_disconnected_shows_offline() -> None:
    state = DashboardState()
    state.apply(DashboardEvent(event_type=EventType.CLIENT_CONNECTED, source="hospital_c"))
    state.apply(DashboardEvent(event_type=EventType.CLIENT_DISCONNECTED, source="hospital_c"))
    assert state.hospitals["hospital_c"].connection_status == "Offline"


def test_metrics_updated_sets_global_metrics() -> None:
    state = DashboardState()
    state.apply(
        DashboardEvent(
            event_type=EventType.METRICS_UPDATED, source="server", round=1,
            payload={"global_dice": 0.4, "global_iou": 0.25, "global_loss": 0.6},
        )
    )
    assert state.global_dice == 0.4
    assert state.global_iou == 0.25
    assert state.global_loss == 0.6


def test_privacy_updated_sets_privacy_fields() -> None:
    state = DashboardState()
    state.apply(
        DashboardEvent(
            event_type=EventType.PRIVACY_UPDATED, source="server", round=1,
            payload={"dp_enabled": True, "epsilon": 0.97, "delta": 1e-5, "budget_status": "ok"},
        )
    )
    assert state.dp_enabled is True
    assert state.epsilon == 0.97
    assert state.budget_status == "ok"


def test_encryption_updated_sets_security_fields() -> None:
    state = DashboardState()
    state.apply(
        DashboardEvent(
            event_type=EventType.ENCRYPTION_UPDATED, source="server",
            payload={"ckks_enabled": True, "tls_status": "Active"},
        )
    )
    assert state.ckks_enabled is True
    assert state.tls_status == "Active"


def test_recent_events_is_bounded() -> None:
    state = DashboardState()
    for i in range(DEFAULT_EVENT_HISTORY_LIMIT + 20):
        state.apply(DashboardEvent(event_type=EventType.SYSTEM_READY, source="server"))
    assert len(state.recent_events) == DEFAULT_EVENT_HISTORY_LIMIT


def test_snapshot_is_fully_json_serializable() -> None:
    import json

    state = DashboardState()
    state.apply(DashboardEvent(event_type=EventType.CLIENT_CONNECTED, source="hospital_a"))
    json.dumps(state.snapshot())  # raises if not serializable


def test_unavailable_metrics_stay_none_not_fabricated() -> None:
    state = DashboardState()
    snapshot = state.snapshot()
    assert snapshot["global_dice"] is None
    assert snapshot["epsilon"] is None
