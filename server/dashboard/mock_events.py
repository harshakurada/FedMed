"""A clearly-isolated synthetic-event generator for frontend development ONLY.

Only used when `DASHBOARD_MOCK_MODE` is explicitly `"true"` (default `"false"` --
`run_dashboard_backend.py` uses real backend events unless this is set). Never imported
by `server/federated/experiment.py` or any other production code path -- this module
exists solely so a frontend developer can see the dashboard react to events without
running any real training.
"""

from __future__ import annotations

import os
import time

from server.dashboard.events import DashboardEvent, EventType
from server.dashboard.websocket_server import DashboardWebSocketServer


def mock_mode_enabled() -> bool:
    return os.environ.get("DASHBOARD_MOCK_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}


def run_mock_event_sequence(server: DashboardWebSocketServer) -> None:
    """A small, clearly-fake sequence -- 2 rounds, 3 hospitals -- for UI development."""
    hospitals = ["hospital_a", "hospital_b", "hospital_c"]

    server.emit(DashboardEvent(event_type=EventType.SYSTEM_READY, source="server"))
    for hospital_id in hospitals:
        server.emit(DashboardEvent(event_type=EventType.CLIENT_CONNECTED, source=hospital_id))

    for round_num in range(1, 3):
        server.emit(
            DashboardEvent(event_type=EventType.ROUND_STARTED, source="server", round=round_num, payload={"num_rounds": 2})
        )
        for hospital_id in hospitals:
            server.emit(DashboardEvent(event_type=EventType.CLIENT_TRAINING, source=hospital_id, round=round_num))
            time.sleep(0.1)
            server.emit(
                DashboardEvent(
                    event_type=EventType.CLIENT_TRAINING_COMPLETED,
                    source=hospital_id,
                    round=round_num,
                    payload={"num_examples": 3, "train_loss": 0.5, "train_dice": 0.3, "train_iou": 0.2},
                )
            )
        server.emit(
            DashboardEvent(
                event_type=EventType.METRICS_UPDATED,
                source="server",
                round=round_num,
                payload={"global_dice": 0.2 + 0.05 * round_num, "global_iou": 0.1 + 0.03 * round_num, "global_loss": 0.7 - 0.05 * round_num},
            )
        )
        server.emit(DashboardEvent(event_type=EventType.GLOBAL_MODEL_UPDATED, source="server", round=round_num))
        server.emit(
            DashboardEvent(
                event_type=EventType.ROUND_COMPLETED,
                source="server",
                round=round_num,
                payload={"round_duration_seconds": 1.5, "clients_completed": 3, "clients_failed": 0},
            )
        )
