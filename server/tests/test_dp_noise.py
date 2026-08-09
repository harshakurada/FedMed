from __future__ import annotations

import numpy as np
import pytest

from server.federated.dp.noise import add_gaussian_noise


def test_noise_is_added_when_dp_enabled() -> None:
    clipped = np.zeros(50)
    rng = np.random.default_rng(0)
    noised = add_gaussian_noise(clipped, clip_norm=1.0, noise_multiplier=1.0, rng=rng)
    assert not np.allclose(noised, clipped)


def test_same_seed_is_reproducible() -> None:
    clipped = np.ones(20)
    noised_1 = add_gaussian_noise(clipped, 1.0, 1.0, np.random.default_rng(42))
    noised_2 = add_gaussian_noise(clipped, 1.0, 1.0, np.random.default_rng(42))
    assert np.array_equal(noised_1, noised_2)


def test_different_seeds_produce_different_noise() -> None:
    clipped = np.ones(20)
    noised_1 = add_gaussian_noise(clipped, 1.0, 1.0, np.random.default_rng(1))
    noised_2 = add_gaussian_noise(clipped, 1.0, 1.0, np.random.default_rng(2))
    assert not np.array_equal(noised_1, noised_2)


def test_noise_scale_grows_with_noise_multiplier_statistically() -> None:
    """Statistical check over many draws, not a hard-coded sample -- empirical std of
    the added noise should scale with noise_multiplier (noise_std = multiplier * clip_norm)."""
    clipped = np.zeros(20000)
    low_noise = add_gaussian_noise(clipped, clip_norm=1.0, noise_multiplier=1.0, rng=np.random.default_rng(0))
    high_noise = add_gaussian_noise(clipped, clip_norm=1.0, noise_multiplier=5.0, rng=np.random.default_rng(0))

    low_std = float(np.std(low_noise))
    high_std = float(np.std(high_noise))
    print(f"empirical std: low_noise_multiplier={low_std:.4f} high_noise_multiplier={high_std:.4f}")
    assert high_std > low_std * 3  # should be ~5x; generous margin for statistical noise
    assert low_std == pytest.approx(1.0, rel=0.1)
    assert high_std == pytest.approx(5.0, rel=0.1)


def test_invalid_noise_multiplier_is_rejected() -> None:
    with pytest.raises(ValueError, match="noise_multiplier"):
        add_gaussian_noise(np.zeros(5), clip_norm=1.0, noise_multiplier=0.0, rng=np.random.default_rng(0))
