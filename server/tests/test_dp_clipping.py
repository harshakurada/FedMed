from __future__ import annotations

import numpy as np
import pytest

from server.federated.dp.clipping import clip_update


def test_vector_below_threshold_is_unchanged() -> None:
    vector = np.array([0.1, 0.2, 0.1])
    result = clip_update(vector, clip_norm=1.0)
    assert np.allclose(result.values, vector)
    assert result.norm_after_clip == pytest.approx(result.norm_before_clip)


def test_vector_above_threshold_is_clipped_to_approximately_c() -> None:
    vector = np.array([3.0, 4.0])  # norm = 5
    result = clip_update(vector, clip_norm=1.0)
    assert result.norm_before_clip == pytest.approx(5.0)
    assert result.norm_after_clip == pytest.approx(1.0)
    assert np.allclose(result.values, np.array([0.6, 0.8]))


def test_zero_vector_does_not_divide_by_zero() -> None:
    result = clip_update(np.zeros(10), clip_norm=1.0)
    assert np.allclose(result.values, 0.0)
    assert result.norm_before_clip == 0.0
    assert result.norm_after_clip == 0.0


def test_very_small_vector_is_numerically_stable() -> None:
    vector = np.full(1000, 1e-12)
    result = clip_update(vector, clip_norm=1.0)
    assert np.all(np.isfinite(result.values))
    assert result.norm_after_clip <= 1.0 + 1e-9


def test_nan_is_rejected_not_silently_propagated() -> None:
    vector = np.array([1.0, float("nan"), 3.0])
    with pytest.raises(ValueError, match="NaN or Inf"):
        clip_update(vector, clip_norm=1.0)


def test_inf_is_rejected_not_silently_propagated() -> None:
    vector = np.array([1.0, float("inf"), 3.0])
    with pytest.raises(ValueError, match="NaN or Inf"):
        clip_update(vector, clip_norm=1.0)


def test_invalid_clip_norm_is_rejected() -> None:
    with pytest.raises(ValueError, match="clip_norm"):
        clip_update(np.array([1.0]), clip_norm=0.0)
    with pytest.raises(ValueError, match="clip_norm"):
        clip_update(np.array([1.0]), clip_norm=-1.0)
