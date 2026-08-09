"""Calibrated Gaussian noise addition -- the second stage of the DP mechanism, distinct
from clipping (`clipping.py`) and from privacy accounting (`accountant.py`).

Noise std = `noise_multiplier * clip_norm`: since clipping already bounds the L2
*sensitivity* of the clipped update to `clip_norm`, calibrating the absolute noise std as
a multiple of `clip_norm` is what makes `accountant.py`'s epsilon formula (which depends
only on `noise_multiplier` and `delta` -- the clip norm cancels out) correct.

`rng` is always an explicit parameter, never a module-global or hard-coded seed: tests
pass a seeded `np.random.default_rng(seed)` for reproducibility; production entry points
construct an unseeded `np.random.default_rng()`. The two code paths never overlap, so a
fixed seed can never accidentally end up in a real experiment.
"""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(
    clipped_values: np.ndarray, clip_norm: float, noise_multiplier: float, rng: np.random.Generator
) -> np.ndarray:
    if noise_multiplier <= 0.0:
        raise ValueError(f"noise_multiplier must be > 0, got {noise_multiplier}")
    noise_std = noise_multiplier * clip_norm
    noise = rng.normal(loc=0.0, scale=noise_std, size=clipped_values.shape)
    return clipped_values + noise
