"""Deterministic, order-preserving chunking of a flattened parameter array into
CKKS-slot-sized pieces.

A 3D U-Net has far more parameters than fit in one `CKKSVector` (CKKS slot capacity =
`poly_modulus_degree / 2`, confirmed by testing this project's own installed TenSEAL --
see `server/federated/encrypted/ckks_config.py`). This is the project's own explicit
chunking layer, not TenSEAL's undocumented auto-batching fallback for over-capacity
inputs (which disables further homomorphic operations and isn't relied on here).
"""

from __future__ import annotations

import numpy as np


def chunk_values(values: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]


def unchunk_values(chunks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
