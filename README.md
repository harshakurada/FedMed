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

## Repository Layout

```
FedMed/
├── server/           # Central aggregation server (Flower strategy, FedAvg, gRPC)
├── hospital-nodes/   # Mock hospital client nodes (local training loops)
├── cv-model/         # 3D U-Net (PyTorch/MONAI) model definitions & training scripts
├── encryption/       # TenSEAL homomorphic encryption + differential privacy utilities
├── dashboard/        # React + Recharts training/monitoring dashboard
├── docs/             # Design notes, architecture diagrams, weekly progress
└── scripts/          # Setup/orchestration helper scripts
```

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
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Dashboard (React)
cd dashboard
npm install
npm start
```

## License

TBD
