"""PyTorch Dataset/DataLoader construction for the raw BraTS pipeline.

Ties discovery -> split -> transforms together. Contains no Flower,
TenSEAL, or dashboard code -- centralized training and each hospital node
(Hospital A/B/C) all call `get_dataloaders` the same way, passing whichever
`BraTSRawConfig` points at their own local data.
"""

from __future__ import annotations

from monai.data import CacheDataset, DataLoader, Dataset

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.discovery import DiscoveryResult, StudyRecord, discover_studies
from cv_model.brats.split import SplitResult, split_studies
from cv_model.brats.transforms import get_train_transforms, get_val_transforms


def build_dataset(studies: tuple[StudyRecord, ...], transform, config: BraTSRawConfig):
    """Wrap a tuple of `StudyRecord`s in a MONAI `Dataset`/`CacheDataset` per `config.cache_mode`."""
    manifest = [s.as_manifest_dict() for s in studies]
    if config.cache_mode == "none":
        return Dataset(data=manifest, transform=transform)
    if config.cache_mode == "cache":
        # Caches the *pre-random-augmentation* transformed volumes in RAM across
        # epochs -- speeds up training but costs `cache_rate * num_studies` volumes
        # of RAM (~few hundred MB each at this pipeline's resampled resolution).
        return CacheDataset(data=manifest, transform=transform, cache_rate=config.cache_rate)
    raise ValueError(f"Unknown cache_mode: {config.cache_mode!r} (expected 'none' or 'cache')")


def build_split(config: BraTSRawConfig = None) -> tuple[DiscoveryResult, SplitResult]:
    """Run discovery then patient-level split; returns both so callers can inspect/report on them."""
    config = config or BraTSRawConfig()
    discovery = discover_studies(config)
    split = split_studies(list(discovery.valid), val_fraction=config.val_fraction, seed=config.seed)
    return discovery, split


def get_dataloaders(config: BraTSRawConfig = None) -> tuple[DataLoader, DataLoader]:
    """Build the training and validation DataLoaders for the raw BraTS pipeline.

    Uses `monai.data.DataLoader` (not `torch.utils.data.DataLoader`) because
    `RandCropByPosNegLabeld` yields multiple crops per volume during training;
    MONAI's loader supplies the `list_data_collate` collate function that
    batches that correctly.
    """
    config = config or BraTSRawConfig()
    _, split = build_split(config)

    train_ds = build_dataset(split.train, get_train_transforms(config), config)
    val_ds = build_dataset(split.val, get_val_transforms(config), config)

    # num_workers=0 by default (see BraTSRawConfig) -- MONAI/PyTorch multiprocess
    # DataLoader workers require the `if __name__ == "__main__":` guard on Windows
    # (spawn-based multiprocessing re-imports the launching script); raise it only
    # from a script that has that guard in place.
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    # Validation runs one volume at a time -- ResizeWithPadOrCropd already gives
    # every sample the same shape, but batch_size=1 keeps memory predictable
    # regardless of patch_size.
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, val_loader
