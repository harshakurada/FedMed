from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cv_model.brats.config import BraTSRawConfig
from cv_model.model import build_unet_from_params
from cv_model.params import get_parameters
from cv_model.training.config import TrainingConfig
from server.federated.evaluation import build_centralized_evaluate_fn, weighted_average


def test_weighted_average_weights_by_sample_count() -> None:
    metrics = [(10, {"dice": 0.5}), (10, {"dice": 0.7}), (80, {"dice": 0.9})]
    result = weighted_average(metrics)
    expected = (10 * 0.5 + 10 * 0.7 + 80 * 0.9) / 100
    assert result["dice"] == pytest.approx(expected)


def test_weighted_average_weights_each_key_by_only_its_own_contributors() -> None:
    # Only the second client reports "iou" -- it must not be diluted by the first
    # client's num_examples, which never contributed a value for that key.
    metrics = [(10, {"dice": 0.5}), (30, {"dice": 0.7, "iou": 0.3})]
    result = weighted_average(metrics)
    assert result["dice"] == pytest.approx((10 * 0.5 + 30 * 0.7) / 40)
    assert result["iou"] == pytest.approx(0.3)


def test_weighted_average_empty_returns_empty_dict() -> None:
    assert weighted_average([]) == {}


def test_weighted_average_ignores_non_numeric_and_bool_values() -> None:
    metrics = [(10, {"hospital_id": "hospital_a", "dice": 0.5}), (10, {"hospital_id": "hospital_b", "dice": 0.7})]
    result = weighted_average(metrics)
    assert "hospital_id" not in result
    assert result["dice"] == pytest.approx(0.6)


def test_centralized_evaluate_fn_returns_loss_and_dice_iou(hospital_data_config: BraTSRawConfig, tmp_path: Path) -> None:
    train_config = replace(
        TrainingConfig(), unet_channels=(4, 8), unet_strides=(2,), unet_num_res_units=1, device_preference="cpu"
    )
    evaluate_fn = build_centralized_evaluate_fn(hospital_data_config, train_config)

    model = build_unet_from_params(
        hospital_data_config.in_channels,
        hospital_data_config.out_channels,
        train_config.unet_channels,
        train_config.unet_strides,
        train_config.unet_num_res_units,
        train_config.device,
    )
    result = evaluate_fn(1, get_parameters(model), {})

    assert result is not None
    loss, metrics = result
    assert isinstance(loss, float)
    assert "dice" in metrics and "iou" in metrics
