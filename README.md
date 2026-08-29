# FedMed

**Privacy-Preserving Federated Learning for Medical Image Segmentation**

This is a simulated research/portfolio project (Modules 1–13, complete). **It is not
clinically validated, makes no claim of medical efficacy, and is not HIPAA/GDPR
compliant.** See Section 11 (Limitations) and the disclaimer in `docs/security.md`.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Run the demo (real DP + CKKS + mutual TLS + dashboard, on small synthetic data by default)
.venv\Scripts\python.exe scripts\run_demo.py

# In a second terminal: the dashboard frontend (needs Node.js/npm)
cd dashboard
npm install
npm start
```

Open `http://localhost:3000`. You'll see a **DEMO MODE** badge, 3 hospitals connecting
over real mutual TLS, a real federated round with DP and CKKS applied, and real (if
small-scale) Dice/IoU/loss numbers. Full walkthrough: Section 13.

## Section 1 — Project Overview

**The problem:** hospitals have valuable medical imaging data but cannot simply
centralize sensitive patient data — regulatory constraints (HIPAA/GDPR) and basic patient
trust mean raw MRI scans and labels should never leave the hospital that holds them.

**The solution:** FedMed allows simulated hospitals to collaboratively train a brain
tumor segmentation model while keeping raw medical data local. Each hospital trains on
its own data; only privacy-protected, encrypted model updates ever leave a hospital, over
an authenticated encrypted channel; the server combines those updates without ever seeing
any individual one in the clear.

## Section 2 — Key Features

- 3D medical image segmentation (BraTS brain tumor MRI)
- 3 simulated hospitals, patient-level data partitioning
- Federated Learning via Flower, real `FedAvg`
- Differential Privacy — client-level clip + Gaussian noise + real privacy accounting
- CKKS Homomorphic Encryption (TenSEAL) — server aggregates ciphertexts, never decrypts
  an individual update
- TLS-secured communication, including genuine mutual TLS (hospital identity tied to a
  certificate)
- gRPC transport
- Real-time monitoring: React dashboard, WebSockets, Recharts
- A clearly-labeled DEMO MODE that runs the full real pipeline on synthetic data in
  seconds, alongside a LIVE MODE that runs the same pipeline on real local BraTS2020 data

## Section 3 — Architecture

```
Hospital A     Hospital B     Hospital C
    |              |              |
    v              v              v
        Local 3D U-Net training (MONAI / PyTorch)
                    |
                    v
    Differential Privacy: clip + Gaussian noise (per hospital)
                    |
                    v
    CKKS homomorphic encryption (per hospital's own public context)
                    |
                    v
    TLS / gRPC mutual authentication (hospital identity <-> certificate)
                    |
                    v
            Federated server
                    |
                    v
    Encrypted aggregation (server never decrypts an individual update)
                    |
                    v
    Authorized decrypt (KeyHolder, distinct from the server) -> Global model
                    |
                    v
    Centralized evaluation -> WebSocket events -> React + Recharts Dashboard
```

Full component-by-component map, key-ownership table, and the integration test proving
this entire chain works together: [`docs/architecture.md`](docs/architecture.md).

## Section 4 — Security Model

Three independent mechanisms, each protecting a different layer — none substitutes for
the others:

**TLS:** protects communication in transit. Without it, anyone observing the network
could read model updates. FedMed uses real mutual TLS in its own coordination service
(hospital identity read from the *verified certificate*, never trusted from the request
body) alongside Flower's own gRPC/TLS live-deployment engine.

**CKKS:** protects model updates during encrypted aggregation. The server combines the
three hospitals' updates via homomorphic operations on ciphertext and never decrypts any
individual one — structurally guaranteed (no server-side code path ever constructs a
private CKKS context), verified by a dedicated test. Only the final aggregate is
decrypted, by a component distinct from the server.

**Differential Privacy:** reduces information leakage from the protected update
mechanism. Each hospital clips and adds calibrated Gaussian noise to its own per-round
contribution before anything leaves it — client-level (hospital-level), not
patient-level.

**These mechanisms do not make the system absolutely secure.** The trust model is
honest-but-curious throughout (no protection against a malicious hospital or a malicious/
compromised server), and none of them make any regulatory-compliance claim. Full threat
model: [`docs/security.md`](docs/security.md), [`docs/threat_model.md`](docs/threat_model.md).

## Section 5 — Data

**BraTS** (Brain Tumor Segmentation): multi-modal MRI (FLAIR, T1, T1ce, T2) with expert
segmentation labels for 3 clinically-relevant tumor regions (Tumor Core, Whole Tumor,
Enhancing Tumor). FedMed uses a **locally-supplied** copy — nothing is auto-downloaded.

**The task:** 3D semantic segmentation — label each voxel into the overlapping tumor
regions above.

**Hospital partitioning:** the discovered studies are split into a global train/validation
set first (patient-level, seeded, never re-derived per hospital), then the training share
is partitioned across 3 simulated hospitals, again patient-level — no study's data is
ever split across two hospitals, and partition isolation is verified by a dedicated test.

No raw patient data, MRI file, or study identifier is exposed anywhere in this README or
any file in this repository. Full details, including the exact label convention and how
to point this at your own data: [`docs/dataset.md`](docs/dataset.md).

## Section 6 — Model

A 3D U-Net (MONAI's `UNet`), trained with PyTorch. Production default architecture:
`unet_channels=(16,32,64,128,256)`, 4 downsampling stages, 2 residual units per stage —
4,810,074 parameters against FedMed's real dataset config. Metrics: Dice and IoU
(per-region, macro-averaged across the 3 tumor regions), plus training/validation loss.

**No state-of-the-art performance claim is made.** Section 9's results are from a
9-study development-scale dataset — far too small to produce clinically meaningful
segmentation. See Section 11.

## Section 7 — Federated Training

1. Each hospital trains the shared model architecture locally on its own patient
   partition (`cv_model.training.engine`, unchanged whether run centrally or federated).
2. Each hospital clips and noises its own contribution (Differential Privacy).
3. Each hospital homomorphically encrypts (CKKS) its DP-protected update.
4. Each hospital submits its encrypted update over a mutually-authenticated TLS/gRPC
   channel.
5. The server aggregates the encrypted updates (weighted `FedAvg`, homomorphically —
   never decrypting an individual contribution).
6. A component distinct from the server decrypts only the final aggregate, producing the
   new global model.
7. The global model is evaluated centrally, against a held-out validation set no hospital
   trained on.

Node-failure handling is real: a dropped hospital never blocks a round or receives fake
substitute parameters, and a late response naming an old round is rejected before
aggregation. Full details: [`docs/federated_training.md`](docs/federated_training.md),
[`docs/secure_communication.md`](docs/secure_communication.md).

## Section 8 — Dashboard

A React + Recharts frontend, fed by a Python `websockets` server, shows:

- Hospital status (connecting / training / completed / error) per hospital
- Round status and progress
- Dice, IoU, and loss (training and centralized-evaluation)
- Privacy budget (epsilon, delta, cumulative epsilon, budget status)
- Encryption status (CKKS enabled/aggregating) and TLS status
- Event stream (last 100 events, severity-coded)
- Training/round history charts
- **Mode indicator** (Module 13): a prominent **LIVE MODE** / **DEMO MODE** badge so the
  dashboard can never be mistaken for a real experiment when it isn't one
- **Project status panel** (Module 13): Federated Learning / DP / CKKS / TLS / WebSocket /
  hospital-count status, read from actual reported state — never hard-coded

Every payload field is checked against an explicit allowlist at construction time — a
forbidden field (patient ID, MRI data, secret key, raw ciphertext, ...) cannot reach a
WebSocket message even by accident. Full schema and examples:
[`docs/websocket_events.md`](docs/websocket_events.md),
[`docs/dashboard.md`](docs/dashboard.md).

## Section 9 — Results

All numbers below are from real runs recorded in Module 12 against a local BraTS2020
subset (9 valid studies) — development-scale numbers proving the pipeline mechanics work
end to end, **not a clinical accuracy claim**. Full commands, environment, and
comparability caveats: [`docs/experiments.md`](docs/experiments.md).

| Experiment | Global Dice | Global IoU | Notes |
|---|---|---|---|
| Centralized baseline (dev-scale, 3 epochs, pooled data) | 0.0259 (val) | 0.0134 (val) | Production architecture, 7 train studies |
| Plain FedAvg (1 round) | 0.0182 | 0.0093 | No DP, no encryption |
| DP FedAvg (1 round) | 0.0186 | 0.0095 | epsilon=0.9690 (client-level) |
| DP + CKKS FedAvg (1 round) | 0.0186 | 0.0095 | Same DP arm, homomorphically aggregated — no measurable utility loss vs. plaintext DP |

Any value not actually measured is written as **NOT MEASURED**, never invented — none
were needed for this table; all 4 rows are real.

**Full end-to-end validation:** the complete DP + CKKS + mutual-TLS + dashboard pipeline
was proven to work together in one federated round in Module 12
(`server/tests/test_final_integration.py`), including a hospital-failure scenario. As of
Module 13, this exact composition is also what the demo script
(`server/federated/integrated_round.py`) runs. 237/237 tests pass across the whole
project. Full report: [`docs/final_validation_report.md`](docs/final_validation_report.md).

## Section 10 — Performance

Real, measured numbers only (source noted per row):

| Metric | Value | Source |
|---|---|---|
| Centralized training time (dev-scale, 3 epochs, CPU) | 104.6s total (~35s/epoch) | `docs/experiments.md`, Module 12 run |
| Utility-comparison round time — plain FedAvg aggregation | 0.066s | `docs/experiments.md`, Module 12 run |
| Utility-comparison round time — DP FedAvg aggregation | 0.035s | `docs/experiments.md`, Module 12 run |
| Utility-comparison round time — DP+CKKS FedAvg (encrypt + homomorphic aggregate) | 35.333s | `docs/experiments.md`, Module 12 run |
| CKKS encryption, real 3D U-Net (4,810,074 params), 1 hospital | ≈8.2s | `docs/homomorphic_encryption.md`, `test_ckks_model_compatibility.py` |
| CKKS decryption, real 3D U-Net, 1 hospital | ≈2.6s | `docs/homomorphic_encryption.md`, `test_ckks_model_compatibility.py` |
| CKKS homomorphic aggregation, 3 hospitals (tiny synthetic model) | ≈0.35s | `docs/homomorphic_encryption.md`, smoke test |
| CKKS numerical error vs. plaintext FedAvg | max abs ≈6×10⁻⁸, mean abs ≈4×10⁻⁹, relative ≈8×10⁻⁷ | `docs/homomorphic_encryption.md`, `test_ckks_aggregation.py` |
| CKKS public context size | 465 KB (with Galois/relin keys disabled — 35 MB otherwise) | `docs/homomorphic_encryption.md` |

Device throughout: CPU only (no CUDA available in this environment).

## Section 11 — Limitations

- **Simulated hospitals**, single local machine — not a real multi-institution
  deployment.
- **Research/portfolio implementation** — no independent security review, penetration
  test, or formal cryptographic audit.
- **Computational overhead**: CKKS encryption/aggregation adds real, measurable time
  (Section 10) — this is a genuine cost of the privacy guarantee, not free.
- **CKKS is approximate**, not exact arithmetic — error is small (Section 10) but
  nonzero.
- **DP utility trade-off**: adding DP noise measurably reduces model utility (Section 9)
  — this is the expected privacy/utility trade-off, not a defect.
- **Dataset limitations**: 9 development-scale studies — not enough for a clinically
  meaningful model; the full ~369-study BraTS2020 release would be needed for that.
- **No clinical validation** of any kind.
- **No production deployment** — the dashboard has no authentication or transport
  encryption (documented local development interface only); `flwr run` against a live
  SuperLink/SuperNode deployment hits an unresolved environment-specific connection issue
  on this Windows/Python 3.14 setup during run submission (see
  `docs/secure_communication.md`).
- DP's cumulative-epsilon accounting uses basic (conservative) composition, not a tight
  RDP/moments accountant..

## Section 12 — Running the Project

All commands are Windows PowerShell. (A Linux/macOS equivalent generally only differs in
activating the virtualenv — `source .venv/bin/activate` instead of
`.venv\Scripts\activate` — and is not separately verified here.)

```powershell
# 1. Python environment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Full test suite (no real dataset required -- synthetic fixtures throughout)
.venv\Scripts\python.exe -m pytest -q

# 3. Point at your own local BraTS2020 copy for anything using real data (never
#    auto-downloaded -- see docs/dataset.md)
$env:FEDMED_BRATS_ROOT = "C:\path\to\your\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"

# 4. Dashboard frontend (requires Node.js/npm)
cd dashboard
npm install
npm start
```

## Section 13 — Demo

```powershell
# Demo mode (default): small synthetic (non-medical) data, full real DP+CKKS+TLS pipeline,
# finishes in seconds. Requires openssl on PATH (Git for Windows bundles one).
.venv\Scripts\python.exe scripts\run_demo.py

# Live mode: same pipeline, real local BraTS2020 data (FEDMED_BRATS_ROOT must be set)
.venv\Scripts\python.exe scripts\run_demo.py --live
```

Then, in a second terminal:

```powershell
cd dashboard
npm start
```

Open `http://localhost:3000`. The dashboard's mode badge shows **DEMO MODE** or
**LIVE MODE** so it's never ambiguous which one is running. Everything you'll see —
hospitals connecting over real mutual TLS, DP noise being applied, CKKS encryption,
encrypted aggregation, a decrypted global model, real Dice/IoU/loss — is produced by the
same real code Modules 6–12 built and tested, not a canned animation. The only simulated
element in DEMO MODE is the underlying data (small synthetic MRI-shaped volumes,
generated in-memory) — clearly disclosed in both the console output and the dashboard.

A separate, even lighter option exists purely for frontend development
(`python -m server.dashboard.run_dashboard_backend --mock`) — a canned event sequence
with no real computation at all. It is intentionally distinct from `scripts/run_demo.py`
and should not be used to represent the system's actual capabilities; see
`docs/dashboard.md`.

## Section 14 — Project Structure

```
FedMed/
├── cv_model/         # 3D U-Net (PyTorch/MONAI), BraTS dataset pipeline, centralized training
├── hospital_nodes/   # Independent hospital-node architecture: partitioning, local training
├── server/           # Federated server: FedAvg, gRPC/TLS, CKKS, DP, dashboard backend, demo
├── dashboard/         # React + Recharts real-time monitoring frontend
├── docs/              # Design docs: architecture, security, threat model, experiments, ...
├── scripts/           # Setup/orchestration helper scripts (dev certs, demo entry point)
├── requirements.txt
└── README.md
```

(`encryption/` from Module 1's original scaffold is superseded by `server/federated/
encrypted/` — kept for history, not the pipeline actually used; see
`docs/homomorphic_encryption.md`. Tests live alongside the code they test —
`cv_model/*/tests/`, `hospital_nodes/tests/`, `server/tests/`, `dashboard/src/*.test.jsx`
— rather than a single top-level `tests/` directory.)

> Python package folders use underscores (`cv_model`, `hospital_nodes`), not hyphens,
> since Python cannot `import` a hyphenated module name.

> Every path/setting-bearing module owns a `config.py` with its own dataclass of
> settings, each overridable via `FEDMED_*` environment variables (`DEMO_MODE` is the one
> deliberate exception — see `server/demo/demo_config.py`). No secret or
> environment-specific path is hard-coded at point of use.

## Documentation Index

| Doc | Covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | System-level component map, data flow, key ownership |
| [`docs/security.md`](docs/security.md) | Consolidated threat model, key ownership, audit findings |
| [`docs/threat_model.md`](docs/threat_model.md) | Threat/mitigation table |
| [`docs/websocket_events.md`](docs/websocket_events.md) | WebSocket event schema, example payloads |
| [`docs/security_checklist.md`](docs/security_checklist.md) | Pre-publish security checklist |
| [`docs/experiments.md`](docs/experiments.md) | Benchmark configs, environment, real results |
| [`docs/final_validation_report.md`](docs/final_validation_report.md) | Module 12's full validation run record |
| [`docs/interview_guide.md`](docs/interview_guide.md) | Interview Q&A grounded in the actual implementation |
| [`docs/project_description.md`](docs/project_description.md) | Short descriptions + resume bullets |
| [`docs/dataset.md`](docs/dataset.md) | BraTS pipeline, dataset layout, label convention |
| [`docs/training.md`](docs/training.md) | Centralized training baseline |
| [`docs/hospitals.md`](docs/hospitals.md) | Hospital-node partitioning and independence |
| [`docs/federated_training.md`](docs/federated_training.md) | FedAvg round orchestration |
| [`docs/secure_communication.md`](docs/secure_communication.md) | gRPC, TLS, mTLS, node resilience |
| [`docs/homomorphic_encryption.md`](docs/homomorphic_encryption.md) | CKKS encryption, threat model |
| [`docs/differential_privacy.md`](docs/differential_privacy.md) | DP mechanism, privacy accounting |
| [`docs/dashboard.md`](docs/dashboard.md) | WebSocket event schema, React dashboard |

## Security & Privacy Disclaimer

FedMed is a research/portfolio implementation of federated learning, homomorphic
encryption, differential privacy, and secure transport concepts, built and validated
against a small local dataset subset. **It has not undergone independent security
review, penetration testing, or formal cryptographic audit. It makes no HIPAA, GDPR, or
other regulatory compliance claim, and no claim of clinical validation or medical
efficacy.** Do not use it, as-is, to process real patient data.

## License

TBD
