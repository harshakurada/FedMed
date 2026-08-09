"""Module 11: a monitoring-only observation layer over Modules 7-10.

This package contains NO training, model, encryption, privacy, or federated-aggregation
logic -- it only converts already-computed, already-approved information (round numbers,
hospital status, Dice/IoU/loss, epsilon/delta, TLS/CKKS/DP status) from the real round
loop (`server/federated/experiment.py`) into WebSocket events a React dashboard can
display. See docs/dashboard.md for the full architecture and event schema.
"""
