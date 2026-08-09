# FedMed — Cross-Silo Federated Learning Engine

**Domain:** Privacy-Preserving Machine Learning (PPML) & Healthcare

This is a simulated research/portfolio project (Modules 1–12, complete). **It is not
clinically validated, makes no claim of medical efficacy, and is not HIPAA/GDPR
compliant** — see "Security & Privacy Disclaimer" at the bottom of this page.

## Problem Statement

Training highly accurate ML models for rare diseases requires massive patient datasets.
However, strict data privacy laws (HIPAA/GDPR) prevent hospitals from sharing raw patient
data with centralized servers.

## Solution

Three simulated hospitals collaborate to train a brain tumor segmentation model on MRI
scans without ever sharing raw patient data:

1. Each hospital trains a local 3D U-Net on its own private patient partition.
2. Each hospital clips and adds calibrated Gaussian noise to its own contribution
   (client-level Differential Privacy) before anything leaves it.
3. Each hospital homomorphically encrypts (CKKS) its DP-protected update and submits it
   over a mutually-authenticated TLS channel.
4. The central server aggregates the three encrypted updates **without ever decrypting
   any individual one** — only the final aggregate is decrypted, by a party distinct from
   the server.
5. Progress streams live to a React dashboard via WebSocket, from an explicit
   payload-safety allowlist that structurally cannot carry raw data or secrets.

## Architecture

```
BraTS2020 -> preprocessing -> patient-level hospital partitioning
   -> 3 Hospital Nodes (local 3D U-Net training)
   -> per-hospital Differential Privacy (clip + noise)
   -> per-hospital CKKS homomorphic encryption
   -> mutual-TLS gRPC transport
   -> homomorphic FedAvg aggregation (server never decrypts an individual update)
   -> authorized decrypt -> global model -> centralized evaluation
   -> WebSocket dashboard events -> React + Recharts UI
```

Full component-by-component map, key-ownership table, and the Module 12 integration test
that proves this entire chain works together: [`docs/architecture.md`](docs/architecture.md).

## Technologies

| Module | Stack | Description |
|---|---|---|
| **Federated Learning Framework** | Flower (`flwr`) | Real `FedAvg` orchestration across 3 hospital nodes, plus Flower's own gRPC/TLS live deployment engine |
| **Computer Vision Model** | PyTorch / MONAI | 3D U-Net for MRI tumor segmentation |
| **Secure Transport** | gRPC + mutual TLS | Hospital identity tied to a client certificate; a small custom coordination service, since Flower's own transport only supports one-way TLS |
| **Homomorphic Encryption** | TenSEAL (CKKS) | Server aggregates ciphertexts; never decrypts an individual hospital's update |
| **Differential Privacy** | NumPy (from first principles) | Client-level clip + Gaussian noise + real privacy accounting; no Opacus/TF-Privacy/PySyft |
| **Monitoring Dashboard** | Python `websockets` + React / Recharts | Real-time, read-only observation layer; no training/model/crypto logic of its own |

## Model

A 3D U-Net (MONAI), Dice + IoU metrics, trained/evaluated identically whether centralized
or federated so the two are actually comparable. Production default architecture:
`unet_channels=(16,32,64,128,256)`, 4 downsampling stages, 2 residual units per stage —
verified against the real FedMed dataset at 4,810,074 parameters
(`docs/homomorphic_encryption.md`'s performance section).

## Dataset

A **locally-supplied** BraTS dataset pipeline (never auto-downloaded): discovery, deep
pixel-level validation, patient-level train/val split, MONAI transforms. Verified
end-to-end against a real local BraTS2020 copy (9 usable studies — enough to prove the
pipeline, not enough for a clinically meaningful model; the full ~369-study release would
be needed for that) plus a synthetic-fixture test suite. Full details, including the
exact label convention and how to point this at your own data:
[`docs/dataset.md`](docs/dataset.md).

## Federated Learning

Real Flower `FedAvg` (weighted by each hospital's sample count) over 3 independent
`HospitalNode`s, with centralized evaluation against a held-out global validation set,
round history/checkpoints/convergence plots, and node-failure/stale-update handling — a
dropped hospital never blocks a round or receives fake substitute parameters,
and a late response naming an old round is rejected before aggregation. Full details:
[`docs/federated_training.md`](docs/federated_training.md), resilience proof in
`server/tests/test_federated_resilience.py`.

## Security

Mutual TLS via a small custom gRPC coordination service (hospital identity read from the
*verified certificate*, never trusted from the request body) alongside Flower's own
gRPC+TLS live deployment engine for FedAvg's actual traffic. Full threat model, consolidated
key-ownership table, and the test-file-to-security-property mapping:
[`docs/security.md`](docs/security.md); transport-layer detail:
[`docs/secure_communication.md`](docs/secure_communication.md).

## Privacy

Client-level (hospital-level, explicitly **not** patient-level — see
[`docs/differential_privacy.md`](docs/differential_privacy.md) for why) Differential
Privacy: each hospital clips and adds calibrated Gaussian noise to its own per-round
contribution before anything leaves it. A real `PrivacyAccountant` derives epsilon from
the classical Gaussian mechanism bound and never resets between rounds; optional budget
enforcement stops further rounds rather than silently exceeding a configured limit.

## Encryption

TenSEAL CKKS homomorphic encryption: the aggregation server combines the 3 hospitals'
updates via real homomorphic addition/weighted-sum on ciphertexts and **never decrypts an
individual update** — structurally guaranteed (no code path in the server-side classes
ever constructs a private CKKS context), verified by a dedicated test, not just
documented. Only the final aggregate is decrypted, by a `KeyHolder` distinct from the
server. Measured numerical error vs. plaintext FedAvg: max abs error ≈ 6×10⁻⁸ — see
[`docs/homomorphic_encryption.md`](docs/homomorphic_encryption.md) for the full threat
model and key-ownership table.

## Monitoring

A Python `websockets` server bridges the real round loop into WebSocket events via an
explicit payload **allowlist** — a forbidden field (patient ID, MRI data, secret key,
raw ciphertext, ...) cannot reach a message even by accident, enforced at event
construction, not filtered later. A React + Recharts frontend shows hospital status,
round progress, Dice/IoU/loss/privacy-budget/encryption/TLS panels, with automatic
reconnection and no fabricated metrics (unavailable values show "N/A"). Monitoring-only —
no training, model, encryption, or privacy logic lives in this layer. Full details:
[`docs/dashboard.md`](docs/dashboard.md).

## Results

All numbers below are from real runs against a local BraTS2020 subset (9 valid studies),
executed as part of Module 12's benchmarking — development-scale numbers proving the
pipeline mechanics work end to end, not a clinical accuracy claim. Full commands,
environment details, and the comparability caveats between rows:
[`docs/experiments.md`](docs/experiments.md).

| Experiment | Global Dice | Global IoU | Notes |
|---|---|---|---|
| Centralized baseline (dev-scale, 3 epochs, pooled data) | 0.0259 (val) | 0.0134 (val) | Production architecture, 7 train studies |
| Plain FedAvg (1 round) | 0.0182 | 0.0093 | No DP, no encryption |
| DP FedAvg (1 round) | 0.0186 | 0.0095 | epsilon=0.9690 (client-level) |
| DP + CKKS FedAvg (1 round) | 0.0186 | 0.0095 | Same DP arm, homomorphically aggregated — CKKS costs ~35s aggregation time, no measurable utility loss vs. plaintext DP |

**Full end-to-end validation:** the complete DP + CKKS + mutual-TLS gRPC + dashboard
pipeline was proven to work together, in one federated round, for the first time in
Module 12 (`server/tests/test_final_integration.py`), including a hospital-failure
scenario. 233/233 tests pass across the whole project. Full report:
[`docs/final_validation_report.md`](docs/final_validation_report.md).

## Limitations

- Development-scale dataset only (9 local BraTS2020 studies) — not enough for a
  clinically meaningful model; the full ~369-study release would be needed for that.
- `cv_model/dataset.py` — an *alternate*, untested path wrapping MONAI's
  auto-downloading `DecathlonDataset`. Kept from Module 1; not the pipeline actually used.
- Dashboard has no authentication or transport encryption (documented local development
  interface only).
- `flwr run` against a live SuperLink/SuperNode deployment hits an unresolved
  environment-specific connection issue on this Windows/Python 3.14 setup during run
  *submission* (SuperNode↔SuperLink TLS registration itself works) —
  [`docs/secure_communication.md`](docs/secure_communication.md).
- DP's cumulative-epsilon accounting uses basic (conservative) composition, not a tight
  RDP/moments accountant.
- Honest-but-curious threat model throughout — no protection against a malicious
  hospital or a malicious/compromised server. Full threat model:
  [`docs/security.md`](docs/security.md).

## Repository Layout

```
FedMed/
├── server/           # Central aggregation server: FedAvg, gRPC/TLS, CKKS, DP, dashboard backend
├── hospital_nodes/   # Independent hospital-node architecture: partitioning, local training
├── cv_model/         # 3D U-Net (PyTorch/MONAI) model definitions & BraTS data pipeline
├── dashboard/        # React + Recharts real-time monitoring frontend
├── docs/             # Per-module design docs + consolidated architecture/security/experiments docs
└── scripts/          # Setup/orchestration helper scripts (e.g. dev cert generation)
```

> Python package folders use underscores (`cv_model`, `hospital_nodes`), not hyphens,
> since Python cannot `import` a hyphenated module name.

> Every path/setting-bearing module (`cv_model/`, `hospital_nodes/`, `server/`,
> `server/federated/*/`) owns a `config.py` with its own dataclass of settings, each
> overridable via `FEDMED_*` environment variables. No secret or environment-specific
> path is hard-coded at point of use.

## Getting Started

```powershell
# Python environment (server / nodes / cv-model / encryption)
python -m venv .venv          # skip if .venv already exists
.venv\Scripts\activate
pip install -r requirements.txt

# Dashboard (React) -- requires Node.js/npm installed first
cd dashboard
npm install
npm start
```

Point `FEDMED_BRATS_ROOT` at your own local BraTS2020 copy to run anything against real
data — nothing in this project auto-downloads it. See [`docs/dataset.md`](docs/dataset.md).

```powershell
# Full test suite (no real dataset required -- synthetic fixtures throughout)
.venv\Scripts\python.exe -m pytest -q

# The Module 12 end-to-end integration test specifically
.venv\Scripts\python.exe -m pytest server\tests\test_final_integration.py -v
```

## Documentation Index

| Doc | Covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | System-level component map, data flow, key ownership |
| [`docs/security.md`](docs/security.md) | Consolidated threat model, key ownership, audit findings |
| [`docs/experiments.md`](docs/experiments.md) | Benchmark configs, environment, real results |
| [`docs/final_validation_report.md`](docs/final_validation_report.md) | Module 12's full validation run record |
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
