# Project Description (Module 13)

All descriptions below reflect the actual implementation (Modules 1–13) — no unsupported
feature or fabricated metric is claimed. Where a number is used, it's a real, measured
figure from `docs/experiments.md`; nowhere is a fake percentage or accuracy claim used.

## 1-line description

FedMed — a privacy-preserving federated learning system for 3D medical image
segmentation, combining Differential Privacy, CKKS homomorphic encryption, and mutual-TLS
secure communication.

## 50-word description

FedMed is a research/portfolio system that trains a 3D U-Net for brain tumor
segmentation across 3 simulated hospitals using Flower's FedAvg, without centralizing
patient data. Each hospital's update is clipped and noised (Differential Privacy), then
homomorphically encrypted (CKKS) and sent over mutual TLS — the server never sees a
plaintext update.

## 100-word description

FedMed is a research/portfolio federated learning system for privacy-preserving medical
image segmentation. Three simulated hospitals train a 3D U-Net (PyTorch/MONAI) locally on
patient-level-partitioned BraTS MRI data. Before any update leaves a hospital, it's
clipped and given calibrated Gaussian noise (Differential Privacy, client-level), then
homomorphically encrypted with CKKS (TenSEAL) and transmitted over a mutually-
authenticated TLS/gRPC channel. The aggregation server combines all three updates via
Flower's FedAvg, entirely on ciphertext — it never decrypts an individual hospital's
contribution, only the final aggregate. A React + WebSocket dashboard shows the entire
pipeline live: training progress, privacy budget, and encryption/TLS status, with no raw
data ever reaching it.

## GitHub description (short, for the repo's "About" field)

Privacy-preserving federated learning for 3D medical image segmentation — FedAvg + CKKS
homomorphic encryption + Differential Privacy + mutual TLS, with a live React/WebSocket
dashboard. Research/portfolio project (not clinically validated).

## LinkedIn project description

**FedMed — Privacy-Preserving Federated Learning for Medical Imaging**

Built an end-to-end federated learning system where 3 simulated hospitals collaboratively
train a 3D U-Net for brain tumor segmentation (BraTS) without ever centralizing patient
data. Implemented the full privacy/security stack from first principles where it
mattered: client-level Differential Privacy (clip + calibrated Gaussian noise + a real
privacy accountant), CKKS homomorphic encryption (TenSEAL) so the aggregation server
never decrypts an individual hospital's update, and genuine mutual TLS over gRPC for
hospital identity and secure transport. Added a real-time React + WebSocket dashboard
with a structurally-enforced payload allowlist (no raw data or secret can reach it, even
by accident) and an end-to-end test proving all of DP + CKKS + mTLS + the dashboard work
together in a single federated round, including a hospital-failure scenario. Research/
portfolio project — not clinically validated, no HIPAA/GDPR compliance claim.

## Resume bullets

- Built a federated learning pipeline (Flower, FedAvg) training a 3D U-Net (PyTorch/
  MONAI) for brain tumor segmentation across 3 simulated hospitals without centralizing
  patient MRI data, with patient-level data partitioning verified by automated tests.
- Implemented client-level Differential Privacy from first principles (gradient clipping,
  calibrated Gaussian noise, a Gaussian-mechanism privacy accountant with cumulative
  epsilon tracking) and CKKS homomorphic encryption (TenSEAL) so the aggregation server
  structurally never decrypts an individual hospital's model update.
- Designed and implemented a genuine mutual-TLS gRPC service for hospital identity and
  secure transport, verified against a real running server across 8+ automated TLS/mTLS
  tests, plus node-failure and stale-update handling so a dropped hospital never blocks a
  round or receives fake substitute parameters.
- Built a real-time monitoring dashboard (React, Recharts, Python WebSockets) with a
  structurally-enforced payload allowlist preventing any raw data or secret from ever
  reaching a client, and authored an end-to-end integration test proving Differential
  Privacy, homomorphic encryption, mutual TLS, and live dashboard events all function
  together correctly in one federated round.

No percentage-based accuracy claim is used in any bullet above — FedMed's own measured
results (`docs/experiments.md`) are development-scale (9 studies) and explicitly not a
clinical accuracy claim; the bullets describe what was engineered and verified, not a
fabricated performance number.
