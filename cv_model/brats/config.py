"""Configuration for the locally-supplied raw BraTS dataset pipeline.

Every path/setting the discovery, validation, transform, and dataset
modules need lives here, overridable via `FEDMED_BRATS_*` environment
variables -- nothing below hard-codes a machine-specific path. BraTS is
never auto-downloaded: `root` must point at wherever the caller already
extracted their own copy (e.g. a Kaggle BraTS2020 archive).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _env_path(var_name: str, default: str) -> Path:
    return Path(os.environ.get(var_name, default)).expanduser().resolve()


def _env_str(var_name: str, default: str) -> str:
    return os.environ.get(var_name, default)


def _env_int(var_name: str, default: int) -> int:
    return int(os.environ.get(var_name, default))


def _env_float(var_name: str, default: float) -> float:
    return float(os.environ.get(var_name, default))


@dataclass(frozen=True)
class BraTSRawConfig:
    """Immutable configuration for the locally-extracted raw BraTS dataset."""

    # --- Dataset location ---
    # Directory whose immediate subdirectories are one per patient/study
    # (e.g. ".../MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/").
    root: Path = field(default_factory=lambda: _env_path("FEDMED_BRATS_ROOT", "./data/brats2020/raw"))

    # --- Expected file layout ---
    # MRI modality suffixes, matched as "<study_id>_<modality><extension>".
    modalities: tuple[str, ...] = ("flair", "t1", "t1ce", "t2")
    # Segmentation label file suffix, same naming pattern as modalities.
    label_suffix: str = "seg"
    # Both compressed and uncompressed NIfTI show up across BraTS mirrors
    # (this project's Kaggle BraTS2020 copy ships uncompressed `.nii`).
    file_extensions: tuple[str, ...] = (".nii.gz", ".nii")

    # What discovery does when a study is missing a required file: "raise"
    # fails discovery immediately with every incomplete study listed;
    # "exclude" leaves it out of the manifest but still reports it (never
    # silently drops it without surfacing that fact to the caller).
    on_incomplete_study: Literal["raise", "exclude"] = "raise"

    # --- Label values actually present in this dataset (see cv_model/brats/labels.py) ---
    valid_label_values: tuple[int, ...] = (0, 1, 2, 4)

    # --- Train/validation split (patient-level, never per-slice) ---
    val_fraction: float = field(default_factory=lambda: _env_float("FEDMED_BRATS_VAL_FRACTION", 0.2))
    seed: int = field(default_factory=lambda: _env_int("FEDMED_BRATS_SEED", 42))

    # --- Volume geometry (kept aligned with cv_model.config.BraTSConfig so a
    # future model built for one data source works unmodified for the other) ---
    pixdim: tuple[float, float, float] = (1.5, 1.5, 1.5)
    patch_size: tuple[int, int, int] = (128, 128, 64)

    # --- Deep validation (Step: "Data Validation") ---
    # Number of studies to actually load pixel data for (NaN/Inf, label
    # value, shape-consistency checks) rather than just checking file headers.
    validation_sample_size: int = field(
        default_factory=lambda: _env_int("FEDMED_BRATS_VALIDATION_SAMPLE_SIZE", 3)
    )

    # --- DataLoader ---
    batch_size: int = field(default_factory=lambda: _env_int("FEDMED_BRATS_BATCH_SIZE", 2))
    num_workers: int = field(default_factory=lambda: _env_int("FEDMED_BRATS_NUM_WORKERS", 0))

    # --- Caching ("none" = monai.data.Dataset, re-reads+re-transforms every
    # epoch, lowest RAM/disk footprint; "cache" = monai.data.CacheDataset,
    # holds `cache_rate` of the *pre-random-augmentation* transformed volumes
    # in RAM for speed -- only enable once dataset size vs. available RAM has
    # been checked, a full BraTS release will not fit at cache_rate=1.0). ---
    cache_mode: Literal["none", "cache"] = field(
        default_factory=lambda: _env_str("FEDMED_BRATS_CACHE_MODE", "none")  # type: ignore[return-value]
    )
    cache_rate: float = field(default_factory=lambda: _env_float("FEDMED_BRATS_CACHE_RATE", 1.0))

    # 3 overlapping BraTS evaluation regions: Tumor Core, Whole Tumor, Enhancing Tumor.
    out_channels: int = 3

    @property
    def in_channels(self) -> int:
        return len(self.modalities)


DEFAULT_CONFIG = BraTSRawConfig()
