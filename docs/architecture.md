# System Architecture (Modules 12–13)

This is a simulated research/portfolio project. **It is not clinically validated and
makes no HIPAA/GDPR compliance claim** — see `docs/security.md`'s disclaimer.

This page is a system-level map across all prior modules. It cross-references the
detailed per-module docs rather than duplicating them — follow the links for the actual
mechanics, threat models, and test evidence behind each box below.

## Simplified flow

```
Hospital A     Hospital B     Hospital C
     |              |              |
     +------ Local training -------+
                    |
                    v
        Differential Privacy (clip + noise)
                    |
                    v
        CKKS homomorphic encryption
                    |
                    v
              TLS / gRPC
                    |
                    v
            Federated server
                    |
                    v
        Encrypted aggregation
                    |
                    v
              Global model
                    |
                    v
               Dashboard
```

## End-to-end data/control flow

```
BraTS2020 (local copy, never auto-downloaded)                    docs/dataset.md
        |
        v
Module 3/5 preprocessing: discovery -> patient-level train/val split -> MONAI transforms
        |
        v
Module 6 hospital partitioning (patient-level, isolation-verified)      docs/hospitals.md
        |
        +---------------+---------------+
        v               v               v
   Hospital A       Hospital B       Hospital C     (independent HospitalNode, own model)
        |               |               |
        v               v               v
   local 3D U-Net training (cv_model.training.engine, unchanged across every module)
        |               |               |
        v               v               v
   Module 10 DP: clip(delta) + Gaussian noise, PER HOSPITAL, before anything leaves it
        |               |               |                            docs/differential_privacy.md
        v               v               v
   Module 9 CKKS encrypt (hospital's own public-context copy)     docs/homomorphic_encryption.md
        |               |               |
        v               v               v
   Module 8 mTLS gRPC SubmitEncryptedUpdate (identity re-verified from TLS cert, not
        |               |               |    request body)             docs/secure_communication.md
        +---------------+---------------+
                        v
        EncryptedAggregationServer: homomorphic weighted sum
        (never decrypts an individual update -- structurally, no private-context code path)
                        v
        KeyHolder.decrypt_aggregate() -- the ONLY .decrypt() call in this project
                        v
        Global model reconstructed -> centralized evaluation (Module 5/7's held-out
        validation split, never a hospital's own data)
                        v
        Module 11 dashboard events (ROUND_STARTED/CLIENT_TRAINING_COMPLETED/
        PRIVACY_UPDATED/ENCRYPTION_UPDATED/METRICS_UPDATED/ROUND_COMPLETED/...) --
        allowlist-enforced payloads only                              docs/dashboard.md
                        v
        WebSocket -> React + Recharts dashboard (read-only observer, no training/
        model/encryption/privacy logic of its own)
```

Plaintext FedAvg (Module 7, `server/federated/experiment.py`) is a separate, still-fully-
supported path that skips the DP/CKKS steps entirely — it is what Module 11's dashboard
integration test (`server/tests/test_dashboard_experiment_integration.py`) exercises, and
what `server/tests/test_final_integration.py` (Module 12) composes with DP + CKKS + real
mTLS + the dashboard, all at once, for the first time — see "The Module 12 integration
test" below.

## Component map

| Component | Module | Role | Doc |
|---|---|---|---|
| `cv_model/` | 4 | 3D U-Net (MONAI), Dice/IoU loss+metrics | — |
| `cv_model/brats/` | 3/5 | BraTS discovery, validation, patient-level split, transforms | `docs/dataset.md` |
| `cv_model/training/` | 4/5 | Training loop, checkpointing, centralized-baseline experiment layer | `docs/training.md` |
| `hospital_nodes/` | 6 | Framework-independent `HospitalNode` + 3-way partitioning | `docs/hospitals.md` |
| `server/federated/` (`experiment.py`, `strategy.py`, `evaluation.py`) | 7 | Real Flower `FedAvg`, centralized evaluation, round history | `docs/federated_training.md` |
| `server/federated/grpc_service/` | 8 | mTLS coordination service (`HealthCheck`, `SubmitEncryptedUpdate`) | `docs/secure_communication.md` |
| `server/federated/encrypted/` | 9 | CKKS homomorphic aggregation (`KeyHolder`, `EncryptedAggregationServer`) | `docs/homomorphic_encryption.md` |
| `server/federated/dp/` | 10 | Client-level DP: clip, noise, privacy accounting | `docs/differential_privacy.md` |
| `server/dashboard/` + `dashboard/` | 11 | WebSocket event bridge + React/Recharts UI | `docs/dashboard.md` |
| `server/federated/integrated_round.py` | 12/13 | The DP+CKKS+mTLS+dashboard round composition, promoted to production code in Module 13 so `server/tests/test_final_integration.py` and the demo entry point share exactly one implementation | this page + `docs/final_validation_report.md` |
| `server/demo/` + `scripts/run_demo.py` | 13 | Single demo entry point: real DP+CKKS+mTLS+dashboard round, on synthetic (DEMO MODE) or real (LIVE MODE) data | README Section 13 |

## Key-ownership summary (full detail in `docs/security.md` / `docs/homomorphic_encryption.md`)

| Party | Holds | Never holds |
|---|---|---|
| Hospital | Its own raw data (never leaves), a public CKKS context, its own mTLS client cert/key | The CKKS secret key, another hospital's data |
| `EncryptedAggregationServer` | A public CKKS context only | The CKKS secret key (structurally — no code path constructs one) |
| `KeyHolder` | The CKKS secret key | Any raw patient data, any TLS private key |
| gRPC coordination service | Server TLS cert/key, CA cert | The CKKS secret key |
| Dashboard (`server/dashboard/`, `dashboard/`) | Already-computed, allowlisted status/metrics | Raw data, model weights, any secret key, any DP seed |

## Privacy/security layering — what each layer does and does not do

Three independent layers, each documented as not substituting for the others
(`docs/homomorphic_encryption.md`'s "TLS vs. CKKS" section, `docs/differential_privacy.md`'s
"Limitations"):

1. **TLS/mTLS (Module 8)** — protects data in transit only. Once received, the receiver
   can read it.
2. **CKKS (Module 9)** — protects the value from the aggregation server's own operator:
   the server computes on ciphertext and never decrypts an individual contribution. Does
   not protect against what a legitimately decrypted final model reveals.
3. **Differential Privacy (Module 10)** — bounds what one hospital's *own contribution*
   this round could reveal, at the client level (not patient level — see
   `docs/differential_privacy.md`'s "Privacy unit"). Applied *before* CKKS encryption, so
   the noise is baked into the ciphertext, not added afterward.

## The Module 12 integration test

`server/tests/test_final_integration.py` is the first place all of the above run together
in one round: 3 real hospitals train locally, each applies DP to its own update, encrypts
it with CKKS, and submits it over the real mutual-TLS gRPC channel (not the in-process
shortcut every earlier module's own tests use for that step); the server aggregates
homomorphically and a distinct `KeyHolder` decrypts the result; the reconstructed global
model is evaluated with Module 7's real centralized-evaluation function; every step's
progress is observed by a real WebSocket dashboard server and a real WebSocket test
client. A second test in the same file injects a hospital failure during the gRPC
submission step and confirms the round still completes with the other two, with the
dashboard showing the failure. See `docs/final_validation_report.md` for the actual pass/
fail result of this run.

## What Module 12 does not change

No new mechanism was invented for this integration: DP, CKKS, mTLS, and the dashboard
event schema are all exactly the Module 8/9/10/11 implementations, called from a new
orchestration path rather than modified. The one small addition to production code is
`server/federated/encrypted/run_encrypted_round.py` gaining an optional `event_sink`
parameter (default `None`, zero behavior change) so the encrypted round loop can report
its own progress to a dashboard — the same pattern Module 11 already used for the
plaintext path.

## Module 13: demo mode

Module 13 added no new ML/privacy/encryption/federated-learning mechanism — it is
polish, documentation, and a demo entry point on top of an already-complete system.

Two changes to production code:

1. **`server/federated/integrated_round.py`** (new): the round-composition logic Module
   12's integration test proved correct was promoted from test-only code into this
   production module, so it has exactly one implementation. `server/tests/
   test_final_integration.py` and `server/demo/run_demo.py` both call it — no
   duplicated orchestration logic.
2. **`server/demo/`** (new) + **`scripts/run_demo.py`** (new): a single entry point that
   starts a real dashboard WebSocket server, generates real dev mTLS certificates, starts
   a real `EncryptedAggregationServer`-backed gRPC server, and runs one real
   `run_integrated_round` — on small synthetic (non-medical) data by default (`DEMO_MODE`,
   defaulting to `true`), or real local BraTS2020 data with `--live`. The dashboard's
   `SYSTEM_READY` event now carries an honest `mode` field (`"DEMO MODE"` / `"LIVE MODE"`)
   so the frontend can show a prominent, truthful mode badge — a small, additive change
   to `server/dashboard/events.py`'s payload allowlist and `state.py`, not a new
   mechanism.

Everything else Module 13 touched is documentation (this page, `docs/security.md`
already existed; `docs/threat_model.md`, `docs/websocket_events.md`,
`docs/security_checklist.md`, `docs/interview_guide.md`, `docs/project_description.md`
are new) or the frontend's presentation layer (a mode badge and a project-status panel
in `dashboard/src/`, both reading already-existing state — no new event types, no new
computation).
