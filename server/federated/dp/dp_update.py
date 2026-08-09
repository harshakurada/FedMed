"""Orchestrates the DP mechanism: clip -> noise -> reconstruct. The single function every
call site uses (no duplicated clip/noise logic anywhere else in this project).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from server.federated.dp.clipping import clip_update
from server.federated.dp.dp_config import DPConfig
from server.federated.dp.noise import add_gaussian_noise


@dataclass
class DPProtectedUpdate:
    dp_params: np.ndarray  # pre_round_global_params + noised_clipped_delta -- same
    # shape/semantics as cv_model.params.get_parameters()'s flattened output, a drop-in
    # replacement for the raw post-training parameters everywhere downstream.
    delta_norm_before_clip: float
    delta_norm_after_clip: float
    noise_std: float


def apply_dp_mechanism(
    pre_round_params: np.ndarray,
    post_training_params: np.ndarray,
    config: DPConfig,
    rng: np.random.Generator,
) -> DPProtectedUpdate:
    """WHAT is protected: `delta = post_training_params - pre_round_params` -- the
    hospital's own contribution this round, not raw model weights, not per-example
    gradients. WHEN/WHERE: called at the hospital immediately after local training,
    before encryption (see server/federated/encrypted/run_encrypted_round.py)."""
    delta = post_training_params - pre_round_params

    clipped = clip_update(delta, config.clip_norm)
    noised_delta = add_gaussian_noise(clipped.values, config.clip_norm, config.noise_multiplier, rng)

    dp_params = pre_round_params + noised_delta
    return DPProtectedUpdate(
        dp_params=dp_params,
        delta_norm_before_clip=clipped.norm_before_clip,
        delta_norm_after_clip=clipped.norm_after_clip,
        noise_std=config.noise_multiplier * config.clip_norm,
    )
