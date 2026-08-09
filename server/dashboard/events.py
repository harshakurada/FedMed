"""The WebSocket event schema -- the single contract between the backend (Modules 7-10's
real round loop) and the React dashboard.

`ALLOWED_PAYLOAD_KEYS` is an explicit allowlist, not a denylist: `DashboardEvent`
construction rejects any payload key that isn't on this list, so a forbidden field
(patient identifiers, MRI data, secret keys, raw ciphertext, raw model weights...) can
never reach a WebSocket message even by accident -- enforced structurally here, not
merely omitted in the frontend. See `server/tests/test_dashboard_events.py` for the
security-audit test that checks this directly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Protocol


class EventType:
    """The 14 events this project's dashboard supports -- deliberately not open-ended."""

    ROUND_STARTED = "ROUND_STARTED"
    ROUND_COMPLETED = "ROUND_COMPLETED"
    CLIENT_CONNECTED = "CLIENT_CONNECTED"
    CLIENT_DISCONNECTED = "CLIENT_DISCONNECTED"
    CLIENT_TRAINING = "CLIENT_TRAINING"
    CLIENT_TRAINING_COMPLETED = "CLIENT_TRAINING_COMPLETED"
    CLIENT_FAILED = "CLIENT_FAILED"
    GLOBAL_MODEL_UPDATED = "GLOBAL_MODEL_UPDATED"
    METRICS_UPDATED = "METRICS_UPDATED"
    PRIVACY_UPDATED = "PRIVACY_UPDATED"
    ENCRYPTION_UPDATED = "ENCRYPTION_UPDATED"
    SYSTEM_WARNING = "SYSTEM_WARNING"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    SYSTEM_READY = "SYSTEM_READY"


ALL_EVENT_TYPES = frozenset(
    value for name, value in vars(EventType).items() if not name.startswith("_")
)

# Every field any event's payload is allowed to carry. Round/hospital identifiers,
# already-computed utility metrics, already-computed privacy/security *status* -- never
# a value that could reconstruct raw data, a secret, or a model tensor.
ALLOWED_PAYLOAD_KEYS = frozenset(
    {
        # Round / system status
        "mode",  # "LIVE MODE" / "DEMO MODE" / "SIMULATION MODE" -- Module 13, so the
        # dashboard can never be mistaken for a real experiment when it isn't one.
        "num_rounds",
        "clients_participating",
        "clients_completed",
        "clients_failed",
        "round_duration_seconds",
        "round_status",
        "system_status",
        "reason",
        # Hospital / client status
        "hospital_id",
        "hospital_status",
        "num_examples",
        "connection_status",
        # Utility metrics (never a privacy or security field)
        "train_loss",
        "train_dice",
        "train_iou",
        "val_dice",
        "val_iou",
        "global_loss",
        "global_dice",
        "global_iou",
        "client_dice",
        "client_iou",
        # Privacy metrics (Module 10) -- never a seed or secret
        "dp_enabled",
        "privacy_unit",
        "epsilon",
        "delta",
        "clip_norm",
        "noise_multiplier",
        "cumulative_epsilon",
        "budget_status",
        "rounds_accounted",
        # Encryption / security status (Module 8/9) -- never key material
        "ckks_enabled",
        "encryption_status",
        "aggregation_mode",
        "ciphertext_aggregation_status",
        "tls_status",
        "secure_communication_status",
        # Human-readable, non-sensitive
        "message",
        "severity",
    }
)


class DashboardEventError(ValueError):
    """Raised when an event or its payload doesn't conform to the schema -- never
    silently dropped or silently stripped."""


@dataclass(frozen=True)
class DashboardEvent:
    event_type: str
    source: str  # "server" or a hospital_id
    round: int | None = None
    payload: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if self.event_type not in ALL_EVENT_TYPES:
            raise DashboardEventError(f"Unknown event_type {self.event_type!r}; must be one of {sorted(ALL_EVENT_TYPES)}")
        validate_event_payload(self.payload)

    def to_json(self) -> str:
        return json.dumps({"type": "event", "data": asdict(self)})


def validate_event_payload(payload: dict) -> None:
    """Raises `DashboardEventError` naming every disallowed key -- never silently drops
    or strips a forbidden field; the caller must fix what it's sending."""
    disallowed = sorted(set(payload) - ALLOWED_PAYLOAD_KEYS)
    if disallowed:
        raise DashboardEventError(
            f"Payload contains field(s) not on the dashboard allowlist: {disallowed}. "
            "The dashboard event schema only carries round/hospital identifiers, "
            "already-computed utility/privacy/security status -- never raw data, model "
            "tensors, or cryptographic secrets."
        )


class EventSink(Protocol):
    """Decouples event *production* (the round loop) from *transport* (WebSocket) --
    the round loop only ever calls `.emit(event)`, never imports `websockets`."""

    def emit(self, event: DashboardEvent) -> None: ...
