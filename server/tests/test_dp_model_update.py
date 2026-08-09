from __future__ import annotations

import numpy as np
import pytest

from server.federated.dp.dp_config import DPConfig
from server.federated.dp.dp_update import apply_dp_mechanism


def test_apply_dp_mechanism_on_small_deterministic_vector() -> None:
    pre_round = np.array([0.0, 0.0, 0.0, 0.0])
    post_training = np.array([3.0, 4.0, 0.0, 0.0])  # delta norm = 5
    config = DPConfig(enabled=True, clip_norm=1.0, noise_multiplier=1.0, delta=1e-5)
    rng = np.random.default_rng(0)

    result = apply_dp_mechanism(pre_round, post_training, config, rng)

    assert result.delta_norm_before_clip == pytest.approx(5.0)
    assert result.delta_norm_after_clip == pytest.approx(1.0)  # clip bound respected pre-noise
    assert result.dp_params.shape == pre_round.shape
    assert result.dp_params.dtype == np.float64
    assert np.all(np.isfinite(result.dp_params))
    assert result.noise_std == pytest.approx(config.noise_multiplier * config.clip_norm)

    # Noise is present: dp_params != pre_round + clipped_delta (the noise-free version)
    clipped_only = pre_round + (post_training - pre_round) * (1.0 / 5.0)  # scale = min(1, 1/5)
    assert not np.allclose(result.dp_params, clipped_only)


def test_apply_dp_mechanism_below_clip_threshold_still_adds_noise() -> None:
    pre_round = np.zeros(10)
    post_training = np.full(10, 0.01)  # small delta, well under clip_norm=1.0
    config = DPConfig(enabled=True, clip_norm=1.0, noise_multiplier=1.0, delta=1e-5)
    rng = np.random.default_rng(1)

    result = apply_dp_mechanism(pre_round, post_training, config, rng)
    assert result.delta_norm_after_clip == pytest.approx(result.delta_norm_before_clip)  # unclipped
    assert not np.allclose(result.dp_params, post_training)  # but noise still present


def test_apply_dp_mechanism_shape_and_dtype_preserved_for_larger_vector() -> None:
    rng = np.random.default_rng(2)
    pre_round = rng.uniform(-1, 1, size=500)
    post_training = pre_round + rng.uniform(-0.1, 0.1, size=500)
    config = DPConfig(enabled=True, clip_norm=0.5, noise_multiplier=2.0, delta=1e-5)

    result = apply_dp_mechanism(pre_round, post_training, config, np.random.default_rng(3))
    assert result.dp_params.shape == (500,)
    assert not np.any(np.isnan(result.dp_params))
    assert not np.any(np.isinf(result.dp_params))
