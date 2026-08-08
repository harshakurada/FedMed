from __future__ import annotations

import numpy as np
from monai.data import DataLoader

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.dataset import build_dataset, get_dataloaders
from cv_model.brats.discovery import discover_studies
from cv_model.brats.split import split_studies
from cv_model.brats.transforms import get_train_transforms, get_val_transforms


def test_train_transform_output_shape(tiny_config: BraTSRawConfig) -> None:
    studies = discover_studies(tiny_config).valid
    manifest = studies[0].as_manifest_dict()

    transformed = get_train_transforms(tiny_config)(manifest)
    # RandCropByPosNegLabeld(num_samples=2) yields a list of 2 crops.
    assert isinstance(transformed, list) and len(transformed) == 2
    for item in transformed:
        assert tuple(item["image"].shape) == (tiny_config.in_channels, *tiny_config.patch_size)
        assert tuple(item["label"].shape) == (tiny_config.out_channels, *tiny_config.patch_size)


def test_val_transform_is_deterministic(tiny_config: BraTSRawConfig) -> None:
    studies = discover_studies(tiny_config).valid
    manifest = studies[0].as_manifest_dict()

    val_transforms = get_val_transforms(tiny_config)
    first = val_transforms(manifest)
    second = val_transforms(manifest)

    assert tuple(first["image"].shape) == (tiny_config.in_channels, *tiny_config.patch_size)
    assert tuple(first["label"].shape) == (tiny_config.out_channels, *tiny_config.patch_size)
    assert np.allclose(np.asarray(first["image"]), np.asarray(second["image"]))
    assert np.array_equal(np.asarray(first["label"]), np.asarray(second["label"]))


def test_dataset_item_returns_image_and_label(tiny_config: BraTSRawConfig) -> None:
    studies = discover_studies(tiny_config).valid
    split = split_studies(studies, val_fraction=tiny_config.val_fraction, seed=tiny_config.seed)

    val_ds = build_dataset(split.val, get_val_transforms(tiny_config), tiny_config)
    item = val_ds[0]
    assert "image" in item and "label" in item


def test_dataloader_produces_correctly_shaped_batch(tiny_config: BraTSRawConfig) -> None:
    train_loader, val_loader = get_dataloaders(tiny_config)
    assert isinstance(train_loader, DataLoader)

    batch = next(iter(val_loader))
    assert tuple(batch["image"].shape) == (1, tiny_config.in_channels, *tiny_config.patch_size)
    assert tuple(batch["label"].shape) == (1, tiny_config.out_channels, *tiny_config.patch_size)
