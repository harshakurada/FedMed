# FedMed — Cross-Silo Federated Learning Engine

**Domain:** Privacy-Preserving Machine Learning (PPML) & Healthcare

## Problem Statement

Training highly accurate ML models for rare diseases requires massive patient datasets. However, strict data privacy laws (HIPAA/GDPR) prevent hospitals from sharing raw patient data with centralized servers.

## Use Case

Researchers at three different global hospitals collaborate to train a brain tumor segmentation model using MRI scans. Instead of pooling their private data, they deploy FedMed nodes:

1. The central server sends the untrained PyTorch model to each hospital.
2. The models train locally on private data.
3. Only the encrypted weight updates are sent back and aggregated (using Secure MultiParty Computation) to update the global model — preserving absolute patient privacy.

## Key Modules

| Module | Stack | Description |
|---|---|---|
| **Federated Learning Framework** | Flower / PySyft | Orchestrates the decentralized training loop, managing communication between the central server and isolated client nodes. |
| **Computer Vision Model** | PyTorch / MONAI | A 3D U-Net architecture designed for segmenting medical imagery (MRI/CT scans). |
| **Privacy & Encryption** | TenSEAL | Implements Homomorphic Encryption, allowing the central server to aggregate model weights while they remain mathematically encrypted. |
| **Training Dashboard** | React / Recharts | Monitoring UI showing the global model's convergence and accuracy metrics across distributed epochs. |

## Current Development Stage

**Implemented:**
- `cv_model/` — 3D U-Net (MONAI), Dice loss/metric. Verified via `python -m cv_model.sanity_check` (synthetic data, no dataset download).
- `cv_model/brats/` — a **locally-supplied** BraTS dataset pipeline (discovery, deep validation, patient-level train/val split, MONAI transforms, PyTorch/MONAI `Dataset`/`DataLoader`, dev-time slice inspection) verified end-to-end against a real local BraTS2020 copy (9 usable studies) plus a synthetic-fixture test suite (`pytest cv_model/brats/tests`, 20/20 passing). Full details, including the exact label convention and how to point it at your own data: [`docs/dataset.md`](docs/dataset.md). This is separate from `cv_model/dataset.py` (below), which targets a different data source.
- `cv_model/training/` — 3D U-Net centralized training baseline: training/validation loops, Dice+IoU metrics, checkpointing (with resume), optional early stopping/LR scheduling, and a small inference/visual-inspection utility. Also the **experiment layer** (`experiment.py`, `results.py`, `plots.py`, `final_evaluation.py`): config validation, a pre-flight data-leakage check, checkpoint-reproducibility verification, and a `results.json` record designed for later comparison against federated results. Full pipeline (leakage check → train → verify checkpoint → evaluate → results/plots) verified end-to-end against real local BraTS2020 data at small scale (see [`docs/training.md`](docs/training.md)). **The official multi-epoch centralized baseline has not been run yet** — run `python -m cv_model.training.run_baseline` explicitly to produce one.
- `hospital_nodes/` — a framework-independent hospital-node architecture (no Flower import): patient-level 3-way data partitioning (`partition.py`, built on Module 3/5's preserved global train/val split), the `HospitalNode` class (`node.py`, own model instance + local training reusing `cv_model.training` unchanged), and copy-safe model-state utilities (`model_state.py`). Verified against real local BraTS2020 data — 3 independent hospitals (sizes 3/2/2), model independence confirmed (training one hospital never changes another's weights). See [`docs/hospitals.md`](docs/hospitals.md).
- `server/federated/` + `hospital_nodes/client_app.py` — **FedAvg federated training loop**: real Flower `FedAvg` (weighted by each hospital's sample count) over the 3 real `HospitalNode`s, round orchestration run in-process, centralized evaluation against Module 5's held-out global validation set, round history/checkpoints/convergence plots, an automatic comparison against the centralized baseline, and (Module 8) node-failure/stale-update handling — a dropped hospital never blocks a round or gets fake parameters, and a delayed response naming an old round is rejected before aggregation. Run the real thing: `python -m server.federated.run_experiment`. Full details: [`docs/federated_training.md`](docs/federated_training.md).
- `server/federated/grpc_service/` — **secure gRPC + TLS** (Module 8): Flower's own SuperLink/SuperNode deployment engine is real gRPC and now configured with real TLS (its native `--ssl-*`/`--root-certificates` flags) for FedAvg's actual traffic; a small additional genuinely-mutual-TLS gRPC service (`HealthCheck` + Module 9's `SubmitEncryptedUpdate`) proves hospital identity tied to a client certificate — something Flower's own transport doesn't support. Dev certs via `python scripts/generate_dev_certs.py`. Full details, including exact PowerShell commands and TLS troubleshooting: [`docs/secure_communication.md`](docs/secure_communication.md).
- `server/federated/encrypted/` — **TenSEAL CKKS homomorphic encryption** (Module 9): the aggregation server combines the 3 hospitals' model updates via real homomorphic addition/weighted-sum on ciphertexts and **never decrypts an individual update** (structurally guaranteed — the server-side code has no path to a private CKKS context, verified by a dedicated test, not just documented); only the final aggregate is decrypted, by a `KeyHolder` distinct from the server. Entirely additive — Module 7's plaintext FedAvg is untouched and still runs via `python -m server.federated.run_experiment`; run the encrypted path with `python -m server.federated.encrypted.run_encrypted_round`. Real measured numerical error (weighted 3-client aggregation vs. plaintext FedAvg): max abs error ≈ 6×10⁻⁸. Verified end-to-end on the actual FedMed 3D U-Net (4.81M parameters) and over the real mTLS gRPC channel, not just small hand-built tensors. Full details, including the threat model and key-ownership table: [`docs/homomorphic_encryption.md`](docs/homomorphic_encryption.md). Supersedes `encryption/config.py` (Module 1's original CKKS placeholder — never implemented, kept for history).
- `server/federated/dp/` — **Differential Privacy** (Module 10): each hospital clips (L2, bound `C`) and adds calibrated Gaussian noise to its own per-round contribution (`Δ = post_training_params − pre_round_global_params`) *before* Module 9's CKKS encryption — the raw update never leaves the hospital. **Privacy unit: client-level (hospital-level), explicitly not patient-level** — see [`docs/differential_privacy.md`](docs/differential_privacy.md) for why. A real `PrivacyAccountant` derives epsilon from the configured mechanism (classical Gaussian mechanism bound; cumulative budget via basic composition, conservative not tight — no fabricated numbers) and never resets between rounds; optional budget enforcement stops further rounds rather than silently exceeding a configured limit. Entirely additive — Module 7's plaintext FedAvg and Module 9's un-noised encrypted path are both untouched (`dp_config=None` is byte-for-byte Module 9's original behavior). Real measured 3-arm comparison (Plain FedAvg / DP FedAvg / DP+CKKS FedAvg, same hospitals, same evaluation): DP costs real Dice (≈0.233 → ≈0.204 on tiny dev data), CKKS adds negligible extra error (~1e-7) on top of DP's own noise. No Opacus/TF-Privacy/PySyft — clipping, noise, and accounting are implemented from first principles in NumPy.
- Configuration structure for dataset/training paths, hospital identities, federated rounds, gRPC/TLS security, CKKS encryption, DP privacy parameters, and server networking — see `cv_model/config.py`, `cv_model/brats/config.py`, `cv_model/training/config.py`, `hospital_nodes/config.py`, `server/config.py`, `server/federated/config.py`, `server/federated/grpc_service/config.py`, `server/federated/encrypted/ckks_config.py`, `server/federated/dp/dp_config.py`.

**Not yet implemented (planned for future modules):**
- A full centralized baseline training run (only 9 usable local studies right now — enough to verify the pipeline, not to produce a meaningful baseline; the fuller ~369-study BraTS2020 release is needed for that) — the federated experiment has the same current data-scale limit.
- `cv_model/dataset.py` — an *alternate*, untested path wrapping MONAI's auto-downloading `DecathlonDataset` (MSD Task01_BrainTumour release). Kept from Module 1; not the pipeline used by `cv_model/brats/`.
- The React dashboard (`dashboard/package.json` declares the intended deps; no components exist yet, and Node.js/npm are not currently installed on this machine).

## Repository Layout

```
FedMed/
├── server/           # Central aggregation server (Flower ServerApp, FedAvg strategy, config)
├── hospital_nodes/   # Mock hospital client nodes (Flower ClientApp, per-hospital config)
├── cv_model/         # 3D U-Net (PyTorch/MONAI) model definitions & data pipeline
├── encryption/       # TenSEAL homomorphic encryption boundary (config only, not yet implemented)
├── dashboard/        # React + Recharts training/monitoring dashboard (scaffold only)
├── docs/             # Design notes, architecture diagrams, weekly progress
└── scripts/          # Setup/orchestration helper scripts
```

> Python package folders use underscores (`cv_model`, `hospital_nodes`), not hyphens,
> since Python cannot `import` a hyphenated module name.

> Each of `cv_model/`, `hospital_nodes/`, `server/`, and `encryption/` owns a `config.py`
> with its own dataclass of settings (paths, ports, hyperparameters), each overridable via
> `FEDMED_*` environment variables. No secrets or environment-specific paths are hard-coded
> at point of use.

## Week-wise Development Plan

### Week 1
- **PPML Engineering:** Centralized Baseline — train a standard 3D U-Net model on a public MRI dataset (e.g., BraTS) to establish a baseline accuracy metric.
- **Distributed Systems:** Node Scaffolding — set up the Flower framework; configure 3 distinct mock "Hospital Nodes" running on separate local ports.

### Week 2
- **PPML Engineering:** Federated Training Loop — partition the dataset across the 3 nodes; implement server logic to broadcast weights, wait for local training, and aggregate results (FedAvg).
- **Distributed Systems:** Secure Communication — implement gRPC with TLS certificates to secure traffic between the server and nodes.

**Mid-Project Review**
- **Federated Audit:** Prove the federated model converges and approaches the accuracy of the centralized baseline without ever exposing raw data to the central server.
- **Node Resilience:** Ensure the training round survives if one of the 3 hospital nodes drops offline mid-epoch.

### Week 3
- **PPML Engineering:** Homomorphic Encryption — integrate TenSEAL; encrypt PyTorch tensors client-side before sending to the server, requiring the server to aggregate on ciphertext.
- **Distributed Systems:** Live Metrics — stream loss/accuracy metrics from the central aggregator to a WebSocket endpoint.

### Week 4
- **PPML Engineering:** Differential Privacy — add controlled statistical noise to weight updates before transmission, mathematically guaranteeing protection against model inversion attacks.
- **Distributed Systems:** Refine & Polish — build the React dashboard to visualize the training loss curve and final MRI tumor segmentation masks.

**Final Review**
A masterclass in cryptography and decentralized deep learning — a compliant, privacy-first healthcare AI architecture.

## Getting Started

```bash
# Python environment (server / nodes / cv-model / encryption)
python -m venv .venv          # skip if .venv already exists
.venv\Scripts\activate
pip install -r requirements.txt

# Dashboard (React) -- requires Node.js/npm installed first
cd dashboard
npm install
npm start
```

## License

TBD
