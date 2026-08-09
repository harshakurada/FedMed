# Real-Time Federated Learning Monitoring Dashboard (Module 11)

This is a simulated research/portfolio project. **The dashboard is a local development
monitoring interface — it performs no training, model, encryption, or privacy
computation itself, and has no authentication (see "Security & authentication" below).**

## Dashboard architecture

```
server/federated/experiment.py's round loop (Module 7, real, unmodified except for one
   optional event_sink hook)
        |  DashboardEvent (event_type, timestamp, round, source, payload -- payload
        |  built from an explicit ALLOWLIST, never raw model/patient data)
        v
server/dashboard/events.py  ->  server/dashboard/state.py (bounded in-memory snapshot)
        v
server/dashboard/websocket_server.py  (Python `websockets` library -- already an
        |                              approved, declared dependency since Module 1;
        |                              not Socket.IO)
        v
dashboard/src/hooks/useDashboardSocket.js  (plain browser WebSocket API)
        v
dashboard/src/App.jsx  ->  KpiCards / HospitalStatus / RoundStatus / MetricsCharts
                            (Recharts) / PrivacyPanel / EncryptionPanel /
                            SecurityStatus / EventLog
```

No new backend or frontend framework, no second Flower server, no second metrics
system — this is purely an observation layer on top of Modules 7–10. The round loop's
actual logic is untouched; it optionally reports what it's already doing.

## WebSocket event schema

Every event (`server/dashboard/events.py::DashboardEvent`) has:

| Field | Meaning |
|---|---|
| `event_type` | One of the 14 types below |
| `source` | `"server"` or a hospital id |
| `round` | Round number, or `null` |
| `payload` | A dict whose keys must all be on the allowlist (below) |
| `timestamp` | ISO-8601 UTC |

| Event | Emitted when |
|---|---|
| `SYSTEM_READY` | Backend/WebSocket server has started |
| `ROUND_STARTED` | A new federated round begins |
| `ROUND_COMPLETED` | A round finishes (aggregation + centralized eval done) |
| `CLIENT_CONNECTED` | A hospital is registered with the round orchestrator |
| `CLIENT_DISCONNECTED` | A hospital disconnects (not currently emitted by the round loop itself — reserved for a future live-deployment bridge; the round loop reports failures via `CLIENT_FAILED`) |
| `CLIENT_TRAINING` | A hospital's local `fit()` call has started |
| `CLIENT_TRAINING_COMPLETED` | A hospital's local `fit()` succeeded |
| `CLIENT_FAILED` | A hospital's `fit()` raised (Module 8's resilience path) |
| `GLOBAL_MODEL_UPDATED` | FedAvg aggregation produced a new global model |
| `METRICS_UPDATED` | Centralized global Dice/IoU/loss available |
| `PRIVACY_UPDATED` | DP accounting produced new epsilon/budget status (Module 10) |
| `ENCRYPTION_UPDATED` | CKKS/TLS status changed (Module 8/9) |
| `SYSTEM_WARNING` | A non-fatal condition worth surfacing |
| `SYSTEM_ERROR` | A fatal condition |

**Payload allowlist** (`ALLOWED_PAYLOAD_KEYS`): round/hospital identifiers,
already-computed utility metrics (loss/Dice/IoU), already-computed privacy metrics
(epsilon/delta/clip_norm/noise_multiplier/budget_status), already-computed security
status (tls_status/ckks_enabled/encryption_status), counts, durations, and short
human-readable messages. Constructing a `DashboardEvent` with any other key raises
`DashboardEventError` — this is enforced at construction time, not filtered later, so a
forbidden field can never reach a WebSocket message even by accident. See
`server/tests/test_dashboard_events.py` for the security-audit test that checks this
directly against every explicitly-forbidden field (`patient_id`, `patient_name`, `MRI`,
`image_data`, `mask`, `raw_model_weights`, `gradient`, `secret_key`, `private_key`,
`ckks_secret`, `dp_seed`, `raw_ciphertext`).

## Dashboard components (`dashboard/src/`)

`App.jsx` composes: `Header` (title + connection status), `KpiCards` (round/Dice/IoU/
loss/epsilon/encryption/hospitals/system-status), `HospitalStatus` (per-hospital cards),
`RoundStatus` (progress bar + counts), `MetricsCharts` (6 Recharts panels: Dice, IoU,
Loss, Privacy budget, Round duration, Client participation — all vs. round), `PrivacyPanel`,
`EncryptionPanel`, `SecurityStatus` (TLS/CKKS/DP active/inactive summary), `EventLog`
(last 100 events, severity-coded). `useDashboardSocket` (`src/hooks/`) is the single
WebSocket connection every component reads from via props — no Redux, no second state
store.

## Security & data restrictions

The dashboard never receives (structurally, per the allowlist above, not by frontend
convention): MRI data, segmentation masks, patient identifiers, raw model weights/
gradients, CKKS secret keys, TLS private keys, DP random seeds, or plaintext client
updates. `server/dashboard/` never imports TenSEAL, never touches a TLS private key, and
never calls into the DP noise mechanism — verified in
`server/tests/test_dashboard_module_compatibility.py`.

## Authentication

**Not implemented.** The existing FedMed specification does not require dashboard
authentication, and this module does not add one — the WebSocket server accepts any
local connection. This is explicitly a local development monitoring interface, not a
production-deployable one; do not expose `server/dashboard/run_dashboard_backend.py`'s
port beyond localhost without adding real authentication first.

## How to start the backend

```powershell
.venv\Scripts\activate
.venv\Scripts\python.exe -m server.dashboard.run_dashboard_backend
```

Runs a small, fast, **real** federated round (tiny synthetic fixtures — the same pattern
Modules 7–10's own tests use) with live events wired to the WebSocket server on
`ws://127.0.0.1:8765`. Not a placeholder: the Dice/IoU/loss/round events you'll see are
genuinely computed by Module 7's real round loop.

## How to start the dashboard

```powershell
cd dashboard
npm install
npm start
```

Opens at `http://localhost:3000`, connecting to `ws://127.0.0.1:8765` by default
(override with `REACT_APP_DASHBOARD_WS_URL` in `dashboard/.env.local`, gitignored).

## Mock mode

```powershell
$env:DASHBOARD_MOCK_MODE = "true"
.venv\Scripts\python.exe -m server.dashboard.run_dashboard_backend
# or, without the env var:
.venv\Scripts\python.exe -m server.dashboard.run_dashboard_backend --mock
```

Default is **disabled** (`DASHBOARD_MOCK_MODE` unset or `"false"`) — real backend events
are used unless mock mode is explicitly turned on. `server/dashboard/mock_events.py` is a
clearly isolated module never imported by production event-emitting code
(`server/federated/experiment.py`).

## How real-time updates work

On connect, `websocket_server.py` sends `{"type": "snapshot", "data": <full current
state>}` immediately — a dashboard never waits for a future event to show current state.
Every subsequent `DashboardEvent` is broadcast as `{"type": "event", "data": ...}` to all
connected clients. The frontend applies each event to its local state in one pass
(`useDashboardSocket.js::applyEvent`, mirroring `server/dashboard/state.py::apply` field
for field) — no full-page reload, no polling.

## How node failures are displayed

When a hospital's `fit()` raises (Module 8's resilience mechanism), the round loop emits
`CLIENT_FAILED`; the dashboard shows that hospital's status as **Error** and logs
"`<hospital> failed to complete round`" in the event log. The round itself continues
with the remaining hospitals (Module 8's own behavior, unchanged) — the next
`ROUND_COMPLETED` event still arrives, and if the same hospital succeeds in a later
round, its status updates to **Completed** again (see
`server/tests/test_dashboard_experiment_integration.py`, which exercises exactly this
sequence against a real round).

## WebSocket troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Dashboard stuck on "Connecting…" | Backend not running, or wrong URL | Start `server.dashboard.run_dashboard_backend`; check `REACT_APP_DASHBOARD_WS_URL` |
| "Disconnected — Attempting to reconnect…" repeating | Backend stopped mid-session | Restart the backend; the dashboard reconnects automatically (exponential backoff, capped at 30s) |
| Metrics/panels show "N/A" | No event has reported that field yet | Expected before the first relevant event (e.g. `METRICS_UPDATED`) arrives — never fabricated |
| `npm start` fails to resolve `react-scripts` | `npm install` wasn't run, or was run before `react-scripts` was added to `package.json` | Re-run `npm install` in `dashboard/` |

## Running the tests

```powershell
# Backend (Python)
.venv\Scripts\python.exe -m pytest server\tests -k dashboard -v

# Frontend (Jest via react-scripts, React Testing Library)
cd dashboard
$env:CI = "true"
npm test -- --watchAll=false
```
