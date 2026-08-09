from __future__ import annotations

from dataclasses import replace

from cv_model.training.config import TrainingConfig
from hospital_nodes.simulation import create_hospital_nodes


def _tiny_train_config() -> TrainingConfig:
    # Note: checkpoint_dir is not set here -- build_hospital_training_config always
    # overrides it to each hospital's own directory regardless. Fine for this test,
    # which only constructs nodes and never calls .fit() (no checkpoint is written).
    return replace(
        TrainingConfig(),
        unet_channels=(4, 8),
        unet_strides=(2,),
        unet_num_res_units=1,
        device_preference="cpu",
    )


def test_create_hospital_nodes_builds_three_independent_non_overlapping_nodes(hospital_data_config) -> None:
    nodes, split = create_hospital_nodes(hospital_data_config, base_train_config=_tiny_train_config(), local_epochs=1)

    assert len(nodes) == 3
    assert len({id(node.model) for node in nodes}) == 3  # 3 distinct model objects

    partition_ids = [{s.study_id for s in node._train_studies} for node in nodes]
    assert partition_ids[0].isdisjoint(partition_ids[1])
    assert partition_ids[1].isdisjoint(partition_ids[2])
    assert partition_ids[0].isdisjoint(partition_ids[2])

    # The global validation set (Module 3/5's holdout) is disjoint from every hospital's
    # local training partition -- no validation study becomes local training data.
    val_ids = {s.study_id for s in split.val}
    for ids in partition_ids:
        assert ids.isdisjoint(val_ids)

    # Every training study is accounted for across the 3 hospitals -- none silently dropped.
    train_ids = {s.study_id for s in split.train}
    assert set().union(*partition_ids) == train_ids
