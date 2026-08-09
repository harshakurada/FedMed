"""Event schema tests, including the required security audit: constructing an event
with any forbidden field must raise -- never be silently stripped."""

from __future__ import annotations

import json

import pytest

from server.dashboard.events import ALL_EVENT_TYPES, DashboardEvent, DashboardEventError, EventType, validate_event_payload

FORBIDDEN_FIELDS = [
    "patient_id",
    "patient_name",
    "MRI",
    "image_data",
    "mask",
    "raw_model_weights",
    "gradient",
    "secret_key",
    "private_key",
    "ckks_secret",
    "dp_seed",
    "raw_ciphertext",
]


def test_valid_event_constructs_and_serializes_to_json() -> None:
    event = DashboardEvent(event_type=EventType.ROUND_STARTED, source="server", round=1, payload={"num_rounds": 3})
    parsed = json.loads(event.to_json())
    assert parsed["type"] == "event"
    assert parsed["data"]["event_type"] == "ROUND_STARTED"
    assert parsed["data"]["round"] == 1


def test_unknown_event_type_is_rejected() -> None:
    with pytest.raises(DashboardEventError):
        DashboardEvent(event_type="NOT_A_REAL_EVENT", source="server")


@pytest.mark.parametrize("forbidden_field", FORBIDDEN_FIELDS)
def test_security_audit_forbidden_field_is_rejected_not_silently_stripped(forbidden_field: str) -> None:
    with pytest.raises(DashboardEventError):
        DashboardEvent(event_type=EventType.METRICS_UPDATED, source="server", payload={forbidden_field: "x"})
    with pytest.raises(DashboardEventError):
        validate_event_payload({forbidden_field: "x"})


def test_all_14_event_types_are_supported() -> None:
    assert len(ALL_EVENT_TYPES) == 14


def test_empty_payload_is_always_valid() -> None:
    validate_event_payload({})


def test_allowed_utility_and_privacy_fields_coexist_without_conflation() -> None:
    # Sanity check that legitimate utility + privacy + security fields are all
    # simultaneously allowed in one event without needing separate schemas.
    event = DashboardEvent(
        event_type=EventType.METRICS_UPDATED,
        source="server",
        round=2,
        payload={"global_dice": 0.5, "epsilon": 0.97, "tls_status": "Active"},
    )
    assert event.payload["global_dice"] == 0.5
