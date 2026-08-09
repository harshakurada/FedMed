from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.discovery import discover_studies
from hospital_nodes.model_state import states_equal
from hospital_nodes.node import HospitalNode
from hospital_nodes.partition import partition_studies
from hospital_nodes.tests.conftest import tiny_hospital_training_config


def _make_three_nodes(
    hospital_data_config: BraTSRawConfig, tmp_path: Path, local_val_fraction: float = 0.0
) -> list[HospitalNode]:
    studies = discover_studies(hospital_data_config).valid
    partitions = partition_studies(list(studies), num_partitions=3, seed=hospital_data_config.seed)
    return [
        HospitalNode(
            studies=partitions[i],
            data_config=hospital_data_config,
            training_config=tiny_hospital_training_config(i, tmp_path, local_val_fraction=local_val_fraction),
        )
        for i in range(3)
    ]


def test_hospital_node_creation_rejects_empty_partition(hospital_data_config, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        HospitalNode(studies=(), data_config=hospital_data_config, training_config=tiny_hospital_training_config(0, tmp_path))


def test_each_hospital_gets_an_independent_model_object(hospital_data_config, tmp_path: Path) -> None:
    nodes = _make_three_nodes(hospital_data_config, tmp_path)
    assert nodes[0].model is not nodes[1].model
    assert nodes[1].model is not nodes[2].model
    assert nodes[0].model is not nodes[2].model


def test_hospital_partitions_do_not_overlap(hospital_data_config, tmp_path: Path) -> None:
    nodes = _make_three_nodes(hospital_data_config, tmp_path)
    id_sets = [{s.study_id for s in node._train_studies} for node in nodes]
    assert id_sets[0].isdisjoint(id_sets[1])
    assert id_sets[1].isdisjoint(id_sets[2])
    assert id_sets[0].isdisjoint(id_sets[2])
    assert all(len(s) > 0 for s in id_sets)


def test_initial_model_states_can_be_equal_but_objects_independent(hospital_data_config, tmp_path: Path) -> None:
    nodes = _make_three_nodes(hospital_data_config, tmp_path)
    shared_state = nodes[0].get_parameters()
    for node in nodes:
        node.set_parameters(shared_state)

    assert states_equal(nodes[0].get_parameters(), nodes[1].get_parameters())
    assert states_equal(nodes[1].get_parameters(), nodes[2].get_parameters())
    assert nodes[0].model is not nodes[1].model  # equal state, still independent objects


def test_local_training_changes_weights_and_leaves_other_hospitals_untouched(hospital_data_config, tmp_path: Path) -> None:
    nodes = _make_three_nodes(hospital_data_config, tmp_path)
    shared_state = nodes[0].get_parameters()
    for node in nodes:
        node.set_parameters(shared_state)

    before = {node.hospital_id: node.get_parameters() for node in nodes}
    result = nodes[0].fit()  # only Hospital A trains
    after = {node.hospital_id: node.get_parameters() for node in nodes}

    a, b, c = nodes[0].hospital_id, nodes[1].hospital_id, nodes[2].hospital_id
    assert not states_equal(before[a], after[a]), "Hospital A's own weights should change after its local training"
    assert states_equal(before[b], after[b]), "Hospital B must be unaffected by Hospital A's training"
    assert states_equal(before[c], after[c]), "Hospital C must be unaffected by Hospital A's training"
    assert result.num_examples == nodes[0].num_local_train_studies


def test_local_training_result_contains_no_raw_dataset_or_tensor_data(hospital_data_config, tmp_path: Path) -> None:
    nodes = _make_three_nodes(hospital_data_config, tmp_path)
    result = nodes[0].fit()
    for value in result.__dict__.values():
        assert not hasattr(value, "modality_paths"), "LocalTrainingResult must never contain a StudyRecord"
        assert not isinstance(value, torch.Tensor), "LocalTrainingResult must never contain raw tensor data"
    for record in result.epoch_records:
        assert record.hospital_id == nodes[0].hospital_id


def test_get_parameters_returns_a_deep_copy_not_a_live_reference(hospital_data_config, tmp_path: Path) -> None:
    nodes = _make_three_nodes(hospital_data_config, tmp_path)
    state = nodes[0].get_parameters()
    key = next(iter(state))
    state[key] += 1.0  # mutate the returned copy
    assert not torch.equal(state[key], nodes[0].get_parameters()[key])


def test_evaluate_returns_none_without_local_validation_configured(hospital_data_config, tmp_path: Path) -> None:
    nodes = _make_three_nodes(hospital_data_config, tmp_path, local_val_fraction=0.0)
    assert nodes[0].evaluate() is None


def test_evaluate_returns_metrics_when_local_validation_is_configured(hospital_data_config, tmp_path: Path) -> None:
    nodes = _make_three_nodes(hospital_data_config, tmp_path, local_val_fraction=0.3)
    result = nodes[0].evaluate()
    assert result is not None
    assert result.hospital_id == nodes[0].hospital_id
    assert result.num_examples == nodes[0].num_local_val_studies > 0


def test_local_checkpoint_is_saved_under_the_hospitals_own_directory(hospital_data_config, tmp_path: Path) -> None:
    nodes = _make_three_nodes(hospital_data_config, tmp_path)
    nodes[0].fit()
    checkpoint_path = nodes[0].training_config.train_config.checkpoint_dir / "checkpoints" / "latest.pt"
    assert checkpoint_path.exists()
    # And it never lands in a sibling hospital's directory.
    other_dir = nodes[1].training_config.train_config.checkpoint_dir
    assert not (other_dir / "checkpoints" / "latest.pt").exists()
