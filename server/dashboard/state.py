"""In-memory dashboard state: consumes `DashboardEvent`s to maintain the current
snapshot a newly-connected dashboard needs immediately, and a bounded recent-event log.
No database -- this is exactly the "existing backend metrics, no new persistence"
the task asks for; state lives only as long as the backend process.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

from server.dashboard.events import DashboardEvent, EventType

DEFAULT_EVENT_HISTORY_LIMIT = 100


@dataclass
class HospitalStatus:
    hospital_id: str
    connection_status: str = "Offline"  # Online / Training / Completed / Offline / Error
    current_round: int | None = None
    last_update: str | None = None
    num_examples: int | None = None
    train_loss: float | None = None
    train_dice: float | None = None
    train_iou: float | None = None


@dataclass
class DashboardState:
    system_status: str = "Not started"
    current_round: int | None = None
    num_rounds: int | None = None
    round_status: str | None = None

    hospitals: dict = field(default_factory=dict)  # hospital_id -> HospitalStatus

    global_loss: float | None = None
    global_dice: float | None = None
    global_iou: float | None = None

    dp_enabled: bool | None = None
    privacy_unit: str | None = None
    epsilon: float | None = None
    delta: float | None = None
    clip_norm: float | None = None
    noise_multiplier: float | None = None
    cumulative_epsilon: float | None = None
    budget_status: str | None = None

    ckks_enabled: bool | None = None
    encryption_status: str | None = None
    tls_status: str | None = None

    recent_events: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_EVENT_HISTORY_LIMIT))

    def apply(self, event: DashboardEvent) -> None:
        payload = event.payload

        if event.round is not None:
            self.current_round = event.round

        if event.event_type == EventType.SYSTEM_READY:
            self.system_status = "Ready"
        elif event.event_type == EventType.ROUND_STARTED:
            self.system_status = "Training"
            self.round_status = "Training"
            self.num_rounds = payload.get("num_rounds", self.num_rounds)
        elif event.event_type == EventType.ROUND_COMPLETED:
            self.round_status = "Completed"
            self.system_status = "Idle"
        elif event.event_type == EventType.CLIENT_CONNECTED:
            hospital = self._hospital(event.source)
            hospital.connection_status = "Online"
            hospital.last_update = event.timestamp
        elif event.event_type == EventType.CLIENT_DISCONNECTED:
            hospital = self._hospital(event.source)
            hospital.connection_status = "Offline"
            hospital.last_update = event.timestamp
        elif event.event_type == EventType.CLIENT_TRAINING:
            hospital = self._hospital(event.source)
            hospital.connection_status = "Training"
            hospital.current_round = event.round
            hospital.last_update = event.timestamp
        elif event.event_type == EventType.CLIENT_TRAINING_COMPLETED:
            hospital = self._hospital(event.source)
            hospital.connection_status = "Completed"
            hospital.current_round = event.round
            hospital.last_update = event.timestamp
            hospital.num_examples = payload.get("num_examples", hospital.num_examples)
            hospital.train_loss = payload.get("train_loss", hospital.train_loss)
            hospital.train_dice = payload.get("train_dice", hospital.train_dice)
            hospital.train_iou = payload.get("train_iou", hospital.train_iou)
        elif event.event_type == EventType.CLIENT_FAILED:
            hospital = self._hospital(event.source)
            hospital.connection_status = "Error"
            hospital.last_update = event.timestamp
        elif event.event_type == EventType.METRICS_UPDATED:
            self.global_loss = payload.get("global_loss", self.global_loss)
            self.global_dice = payload.get("global_dice", self.global_dice)
            self.global_iou = payload.get("global_iou", self.global_iou)
        elif event.event_type == EventType.PRIVACY_UPDATED:
            self.dp_enabled = payload.get("dp_enabled", self.dp_enabled)
            self.privacy_unit = payload.get("privacy_unit", self.privacy_unit)
            self.epsilon = payload.get("epsilon", self.epsilon)
            self.delta = payload.get("delta", self.delta)
            self.clip_norm = payload.get("clip_norm", self.clip_norm)
            self.noise_multiplier = payload.get("noise_multiplier", self.noise_multiplier)
            self.cumulative_epsilon = payload.get("cumulative_epsilon", self.cumulative_epsilon)
            self.budget_status = payload.get("budget_status", self.budget_status)
        elif event.event_type == EventType.ENCRYPTION_UPDATED:
            self.ckks_enabled = payload.get("ckks_enabled", self.ckks_enabled)
            self.encryption_status = payload.get("encryption_status", self.encryption_status)
            self.tls_status = payload.get("tls_status", self.tls_status)

        self.recent_events.append(asdict(event))

    def _hospital(self, hospital_id: str) -> HospitalStatus:
        if hospital_id not in self.hospitals:
            self.hospitals[hospital_id] = HospitalStatus(hospital_id=hospital_id)
        return self.hospitals[hospital_id]

    def snapshot(self) -> dict:
        """Everything a newly-connected dashboard needs to render immediately, without
        waiting for a future event."""
        data = asdict(self)
        data["hospitals"] = {hid: asdict(status) for hid, status in self.hospitals.items()}
        data["recent_events"] = list(self.recent_events)
        return data
