# WebSocket Event & API Documentation (Module 13)

The authoritative schema lives in code: `server/dashboard/events.py`
(`EventType`, `ALLOWED_PAYLOAD_KEYS`, `DashboardEvent`) and
`server/dashboard/state.py` (how each event mutates the snapshot). This page documents
that schema for reference — if the two ever disagree, the code is correct.

## Message envelope

Every WebSocket message is one of two shapes:

```json
{"type": "snapshot", "data": { /* full DashboardState, sent once on connect */ }}
{"type": "event", "data": { /* one DashboardEvent, sent as it happens */ }}
```

A `DashboardEvent`:

```json
{
  "event_type": "PRIVACY_UPDATED",
  "source": "hospital_a",
  "round": 1,
  "payload": {"dp_enabled": true, "epsilon": 0.969},
  "timestamp": "2026-08-09T10:15:00.123456+00:00"
}
```

`source` is `"server"` for system/round/aggregation events, or a hospital id
(`hospital_a`/`hospital_b`/`hospital_c`) for per-hospital events. `payload` keys are
checked against an explicit allowlist **at construction time** — an unlisted key raises
`DashboardEventError` before the event can ever be sent (see `docs/security.md`).

## Event types

### `SYSTEM_READY`
**Purpose:** the backend/WebSocket server has started.
**Payload fields:** `mode` (optional, Module 13 — `"LIVE MODE"` / `"DEMO MODE"` /
`"SIMULATION MODE"`).
**Example:**
```json
{"event_type": "SYSTEM_READY", "source": "server", "round": null, "payload": {"mode": "DEMO MODE"}, "timestamp": "2026-08-09T10:00:00Z"}
```
**Security:** no restrictions beyond the general allowlist — this event never carries
data.

### `ROUND_STARTED`
**Purpose:** a new federated round begins.
**Payload fields:** `num_rounds`.
**Example:**
```json
{"event_type": "ROUND_STARTED", "source": "server", "round": 1, "payload": {"num_rounds": 3}, "timestamp": "2026-08-09T10:00:01Z"}
```

### `ROUND_COMPLETED`
**Purpose:** a round finishes (aggregation + evaluation done).
**Payload fields:** `round_duration_seconds`, `clients_completed`, `clients_failed`,
`round_status`.
**Example:**
```json
{"event_type": "ROUND_COMPLETED", "source": "server", "round": 1, "payload": {"round_duration_seconds": 1.8, "clients_completed": 3, "clients_failed": 0, "round_status": "Completed"}, "timestamp": "2026-08-09T10:00:05Z"}
```

### `CLIENT_CONNECTED`
**Purpose:** a hospital is registered with the round orchestrator (or, in the Module 13
demo, has just completed a real mTLS `HealthCheck`).
**Payload fields:** none currently used.
**Example:**
```json
{"event_type": "CLIENT_CONNECTED", "source": "hospital_a", "round": null, "payload": {}, "timestamp": "2026-08-09T10:00:00Z"}
```

### `CLIENT_DISCONNECTED`
**Purpose:** reserved for a future live-deployment bridge; the in-process round loop
reports failures via `CLIENT_FAILED` instead. Not currently emitted by any production
path.
**Payload fields:** none currently used.

### `CLIENT_TRAINING`
**Purpose:** a hospital's local `fit()` call has started.
**Payload fields:** none currently used (the hospital and round are already carried by
`source`/`round`).
**Example:**
```json
{"event_type": "CLIENT_TRAINING", "source": "hospital_b", "round": 1, "payload": {}, "timestamp": "2026-08-09T10:00:02Z"}
```

### `CLIENT_TRAINING_COMPLETED`
**Purpose:** a hospital's local `fit()` succeeded.
**Payload fields:** `num_examples`, `train_loss`, `train_dice`, `train_iou`.
**Example:**
```json
{"event_type": "CLIENT_TRAINING_COMPLETED", "source": "hospital_b", "round": 1, "payload": {"num_examples": 3, "train_loss": 0.87, "train_dice": 0.04, "train_iou": 0.02}, "timestamp": "2026-08-09T10:00:03Z"}
```

### `CLIENT_FAILED`
**Purpose:** a hospital's local training or its encrypted-update submission failed
(Module 8/13's resilience path).
**Payload fields:** `reason`.
**Example:**
```json
{"event_type": "CLIENT_FAILED", "source": "hospital_c", "round": 1, "payload": {"reason": "submission_failed"}, "timestamp": "2026-08-09T10:00:04Z"}
```

### `GLOBAL_MODEL_UPDATED`
**Purpose:** aggregation (plaintext or homomorphic) produced a new global model.
**Payload fields:** `aggregation_mode`.
**Example:**
```json
{"event_type": "GLOBAL_MODEL_UPDATED", "source": "server", "round": 1, "payload": {"aggregation_mode": "encrypted (ciphertext)"}, "timestamp": "2026-08-09T10:00:04Z"}
```

### `METRICS_UPDATED`
**Purpose:** centralized global Dice/IoU/loss is available (from real evaluation against
the held-out validation set).
**Payload fields:** `global_loss`, `global_dice`, `global_iou`.
**Example:**
```json
{"event_type": "METRICS_UPDATED", "source": "server", "round": 1, "payload": {"global_loss": 0.77, "global_dice": 0.21, "global_iou": 0.12}, "timestamp": "2026-08-09T10:00:05Z"}
```

### `PRIVACY_UPDATED`
**Purpose:** DP accounting produced a new epsilon/budget status for a hospital.
**Payload fields:** `dp_enabled`, `privacy_unit`, `epsilon`, `delta`, `clip_norm`,
`noise_multiplier`, `cumulative_epsilon`, `budget_status`.
**Example:**
```json
{"event_type": "PRIVACY_UPDATED", "source": "hospital_a", "round": 1, "payload": {"dp_enabled": true, "epsilon": 0.969, "cumulative_epsilon": 0.969, "delta": 1e-5, "clip_norm": 0.5, "noise_multiplier": 5.0}, "timestamp": "2026-08-09T10:00:02Z"}
```
**Security:** never carries `dp_seed` — the RNG seed used for noise (forbidden field,
checked directly by `server/tests/test_dashboard_events.py`).

### `ENCRYPTION_UPDATED`
**Purpose:** CKKS/TLS status changed.
**Payload fields:** `ckks_enabled`, `encryption_status`, `tls_status`,
`aggregation_mode`, `ciphertext_aggregation_status`, `secure_communication_status`.
**Example:**
```json
{"event_type": "ENCRYPTION_UPDATED", "source": "server", "round": null, "payload": {"ckks_enabled": true, "tls_status": "Active", "encryption_status": "context ready"}, "timestamp": "2026-08-09T10:00:01Z"}
```
**Security:** never carries `ckks_secret`, `secret_key`, or `private_key` — the CKKS
secret key and any TLS private key (forbidden fields, checked directly by
`server/tests/test_dashboard_events.py`).

### `SYSTEM_WARNING`
**Purpose:** a non-fatal condition worth surfacing.
**Payload fields:** `message`, `severity`.

### `SYSTEM_ERROR`
**Purpose:** a fatal condition.
**Payload fields:** `message`, `severity`.

## Forbidden fields (never allowed, checked at construction time)

`patient_id`, `patient_name`, `MRI`, `image_data`, `mask`, `raw_model_weights`,
`gradient`, `secret_key`, `private_key`, `ckks_secret`, `dp_seed`, `raw_ciphertext` — and
any other key not on the explicit allowlist in `server/dashboard/events.py`. Constructing
a `DashboardEvent` with any of these raises `DashboardEventError` before the event can be
sent. See `server/tests/test_dashboard_events.py` for the security-audit test that checks
this directly.

## Full schema reference

For the complete, currently-accurate list of every event type and every allowed payload
key, see `server/dashboard/events.py::EventType` and `::ALLOWED_PAYLOAD_KEYS` directly —
this page is documentation, the code is the source of truth.
