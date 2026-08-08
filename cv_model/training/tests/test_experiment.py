from __future__ import annotations

from dataclasses import replace

import pytest

# Reuses Module 3's synthetic-dataset fixtures rather than duplicating the
# nibabel-fixture-writing code for a second test package.
from cv_model.brats.tests.conftest import synthetic_dataset_root, tiny_config  # noqa: F401
from cv_model.training.config import TrainingConfig
from cv_model.training.experiment import (
    DataLeakageError,
    ExperimentConfig,
    ExperimentConfigError,
    check_no_data_leakage,
    validate_experiment_config,
)


def _experiment_config(tiny_config, **train_overrides) -> ExperimentConfig:
    train_config = replace(TrainingConfig(), device_preference="cpu", **train_overrides)
    return ExperimentConfig(name="unit-test", data_config=tiny_config, train_config=train_config)


def test_validate_experiment_config_accepts_reasonable_defaults(tiny_config) -> None:
    validate_experiment_config(_experiment_config(tiny_config))  # should not raise


def test_validate_experiment_config_rejects_zero_epochs(tiny_config) -> None:
    with pytest.raises(ExperimentConfigError, match="epochs"):
        validate_experiment_config(_experiment_config(tiny_config, epochs=0))


def test_validate_experiment_config_rejects_mismatched_strides_and_channels(tiny_config) -> None:
    with pytest.raises(ExperimentConfigError, match="unet_strides"):
        validate_experiment_config(
            _experiment_config(tiny_config, unet_channels=(16, 32, 64), unet_strides=(2, 2, 2))
        )


def test_validate_experiment_config_rejects_wrong_out_channels(tiny_config) -> None:
    bad_data_config = replace(tiny_config, out_channels=1)
    with pytest.raises(ExperimentConfigError, match="out_channels"):
        validate_experiment_config(ExperimentConfig(name="x", data_config=bad_data_config, train_config=TrainingConfig()))


def test_check_no_data_leakage_passes_on_clean_synthetic_dataset(tiny_config) -> None:
    config = _experiment_config(tiny_config)
    check_no_data_leakage(config)  # should not raise


def test_check_no_data_leakage_detects_random_augmentation_in_val_transforms(tiny_config, monkeypatch) -> None:
    import cv_model.training.experiment as experiment_module
    from monai.transforms import Compose, RandFlipd

    def _val_transforms_with_augmentation(config):
        return Compose([RandFlipd(keys=["image"], prob=1.0, spatial_axis=0)])

    monkeypatch.setattr(experiment_module, "get_val_transforms", _val_transforms_with_augmentation)
    with pytest.raises(DataLeakageError, match="random augmentation"):
        check_no_data_leakage(_experiment_config(tiny_config))
