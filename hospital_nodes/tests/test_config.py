from __future__ import annotations

import pytest

from hospital_nodes.config import HOSPITAL_NODES, build_hospital_training_config, get_hospital_config


def test_all_three_hospital_configs_load() -> None:
    assert len(HOSPITAL_NODES) == 3
    for i in range(3):
        config = get_hospital_config(i)
        assert config.partition_id == i


def test_hospital_ids_are_unique() -> None:
    ids = [h.partition_id for h in HOSPITAL_NODES]
    assert len(ids) == len(set(ids))


def test_hospital_names_are_unique() -> None:
    names = [h.name for h in HOSPITAL_NODES]
    assert len(names) == len(set(names))


def test_invalid_hospital_id_fails_clearly() -> None:
    with pytest.raises(ValueError, match="No hospital configured"):
        get_hospital_config(99)


def test_hospital_training_config_checkpoint_dirs_are_unique() -> None:
    configs = [build_hospital_training_config(i, local_epochs=1) for i in range(3)]
    checkpoint_dirs = [c.train_config.checkpoint_dir for c in configs]
    assert len(checkpoint_dirs) == len(set(checkpoint_dirs))


def test_hospital_training_config_defaults_to_no_local_validation() -> None:
    config = build_hospital_training_config(0, local_epochs=1)
    assert config.local_val_fraction == 0.0


def test_hospital_training_config_local_epochs_is_configurable() -> None:
    config = build_hospital_training_config(1, local_epochs=5)
    assert config.local_epochs == 5
