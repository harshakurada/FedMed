from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.discovery import discover_studies
from cv_model.params import get_parameters
from hospital_nodes.client_app import HospitalNodeClient
from hospital_nodes.node import HospitalNode
from hospital_nodes.partition import partition_studies
from hospital_nodes.tests.conftest import tiny_hospital_training_config


def _make_node(
    hospital_data_config: BraTSRawConfig, tmp_path: Path, partition_id: int = 0, local_val_fraction: float = 0.0
) -> HospitalNode:
    studies = discover_studies(hospital_data_config).valid
    partitions = partition_studies(list(studies), num_partitions=3, seed=hospital_data_config.seed)
    return HospitalNode(
        studies=partitions[partition_id],
        data_config=hospital_data_config,
        training_config=tiny_hospital_training_config(partition_id, tmp_path, local_val_fraction=local_val_fraction),
    )


def test_get_parameters_matches_cv_model_params_conversion(hospital_data_config, tmp_path: Path) -> None:
    node = _make_node(hospital_data_config, tmp_path)
    client = HospitalNodeClient(node)
    ndarrays = client.get_parameters({})
    expected = get_parameters(node.model)
    assert len(ndarrays) == len(expected)
    assert all((actual == exp).all() for actual, exp in zip(ndarrays, expected))


def test_fit_changes_weights_and_reports_num_examples(hospital_data_config, tmp_path: Path) -> None:
    node = _make_node(hospital_data_config, tmp_path)
    client = HospitalNodeClient(node)
    # cv_model.params.get_parameters returns numpy views aliasing the model's live
    # tensors (no copy when already on CPU) -- .copy() is required for a snapshot that
    # survives the model being trained in place afterwards.
    before = [arr.copy() for arr in client.get_parameters({})]
    new_params, num_examples, metrics = client.fit(before, {})

    assert num_examples == node.num_local_train_studies
    assert not all((a == b).all() for a, b in zip(new_params, before))
    for key in ("hospital_id", "train_loss", "train_dice", "train_iou"):
        assert key in metrics
    assert "val_dice" not in metrics  # no local validation configured for this hospital
    assert "val_iou" not in metrics


def test_evaluate_raises_without_local_validation_configured(hospital_data_config, tmp_path: Path) -> None:
    node = _make_node(hospital_data_config, tmp_path, local_val_fraction=0.0)
    client = HospitalNodeClient(node)
    params = client.get_parameters({})
    with pytest.raises(NotImplementedError):
        client.evaluate(params, {})


def test_evaluate_returns_metrics_when_local_validation_configured(hospital_data_config, tmp_path: Path) -> None:
    node = _make_node(hospital_data_config, tmp_path, local_val_fraction=0.3)
    client = HospitalNodeClient(node)
    params = client.get_parameters({})
    loss, num_examples, metrics = client.evaluate(params, {})
    assert isinstance(loss, float)
    assert num_examples == node.num_local_val_studies > 0
    assert "dice" in metrics and "iou" in metrics


def test_fit_result_never_contains_raw_data(hospital_data_config, tmp_path: Path) -> None:
    node = _make_node(hospital_data_config, tmp_path)
    client = HospitalNodeClient(node)
    params = client.get_parameters({})
    _new_params, _num_examples, metrics = client.fit(params, {})
    for value in metrics.values():
        assert not isinstance(value, torch.Tensor)
        assert not hasattr(value, "modality_paths")
