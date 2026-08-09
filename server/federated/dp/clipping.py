"""L2-norm clipping of one hospital's per-round update.

**WHAT is clipped:** `delta = post_training_params - pre_round_global_params` -- the
hospital's own contribution this round, computed by the caller (see `dp_update.py`).
NOT the raw post-training model parameters (clipping those would be meaningless -- a
well-trained model's weights aren't "small," and clipping them would just damage the
model) and NOT per-example gradients (this project has no per-example gradient access at
this boundary, and Opacus/TF-Privacy -- which would provide that -- are explicitly out of
scope). This matches the standard DP-FedAvg / client-level-DP construction (McMahan et
al., 2017) and Abadi et al.'s DP-SGD convention of a single global L2 clip over the
entire flattened vector, not per-layer.

**WHERE/WHEN:** at the hospital, immediately after local training completes and before
any noise or encryption (see `dp_update.py::apply_dp_mechanism`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClippedUpdate:
    values: np.ndarray
    norm_before_clip: float
    norm_after_clip: float


def clip_update(delta: np.ndarray, clip_norm: float) -> ClippedUpdate:
    """`delta_clipped = delta * min(1, clip_norm / ||delta||_2)`.

    Safe handling: a zero vector is returned unchanged (no division by zero). NaN/Inf
    anywhere in `delta` raises a clear `ValueError` -- never silently propagated into a
    corrupted model update.
    """
    if clip_norm <= 0.0:
        raise ValueError(f"clip_norm must be > 0, got {clip_norm}")
    if not np.all(np.isfinite(delta)):
        raise ValueError("delta contains NaN or Inf values -- refusing to clip a corrupted update.")

    norm = float(np.linalg.norm(delta))
    if norm == 0.0:
        return ClippedUpdate(values=delta.copy(), norm_before_clip=0.0, norm_after_clip=0.0)

    scale = min(1.0, clip_norm / norm)
    clipped = delta * scale
    return ClippedUpdate(values=clipped, norm_before_clip=norm, norm_after_clip=float(np.linalg.norm(clipped)))
