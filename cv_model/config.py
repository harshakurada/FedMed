"""Configuration for the FedMed centralized 3D U-Net baseline.

Centralizes every path, hyperparameter, and device setting used by the
data pipeline and model so training/inference scripts never hardcode
magic values. Values can be overridden via environment variables to
support different machines (local GPU box, cloud VM, CI) without any
code changes.

Federated modules (Week 2+) reuse this same config so every hospital
node and the central server build architecturally identical models --
required for FedAvg weight aggregation to be valid.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


def _env_path(var_name: str, default: str) -> Path:
    """Resolve a filesystem path from an environment variable, falling back to `default`."""
    return Path(os.environ.get(var_name, default)).expanduser().resolve()


def _env_int(var_name: str, default: int) -> int:
    """Resolve an integer setting from an environment variable, falling back to `default`."""
    return int(os.environ.get(var_name, default))


def _env_float(var_name: str, default: float) -> float:
    """Resolve a float setting from an environment variable, falling back to `default`."""
    return float(os.environ.get(var_name, default))


@dataclass(frozen=True)
class BraTSConfig:
    """Immutable configuration for the BraTS/MSD Task01_BrainTumour pipeline and 3D U-Net."""

    # --- Dataset ---
    # Root directory MONAI's DecathlonDataset downloads/extracts Task01_BrainTumour into.
    data_root: Path = field(default_factory=lambda: _env_path("FEDMED_DATA_ROOT", "./data"))
    # Fraction of the labeled "training" list held out for validation (MONAI splits this
    # internally and reproducibly using `seed`, since Task01_BrainTumour ships no separate
    # labeled validation set).
    val_fraction: float = field(default_factory=lambda: _env_float("FEDMED_VAL_FRACTION", 0.2))
    # Seeds the train/val split so re-running training reproduces the same partition.
    seed: int = field(default_factory=lambda: _env_int("FEDMED_SEED", 42))
    # Directory model checkpoints are written to (created by the training script, not here).
    checkpoint_dir: Path = field(default_factory=lambda: _env_path("FEDMED_CHECKPOINT_DIR", "./checkpoints"))

    # --- Volume geometry ---
    # Isotropic voxel spacing (mm) volumes are resampled to before cropping.
    pixdim: tuple[float, float, float] = (1.5, 1.5, 1.5)
    # Size of the random 3D patch cropped from each volume during training.
    patch_size: tuple[int, int, int] = (128, 128, 64)

    # --- Model ---
    # 4 MRI modalities per BraTS case: FLAIR, T1w, T1gd (post-contrast), T2w.
    in_channels: int = 4
    # 3 overlapping BraTS evaluation regions: Tumor Core, Whole Tumor, Enhancing Tumor.
    out_channels: int = 3
    unet_channels: tuple[int, ...] = (16, 32, 64, 128, 256)
    unet_strides: tuple[int, ...] = (2, 2, 2, 2)
    unet_num_res_units: int = 2

    # --- Training ---
    batch_size: int = field(default_factory=lambda: _env_int("FEDMED_BATCH_SIZE", 2))
    num_workers: int = field(default_factory=lambda: _env_int("FEDMED_NUM_WORKERS", 4))
    learning_rate: float = field(default_factory=lambda: _env_float("FEDMED_LR", 1e-4))
    max_epochs: int = field(default_factory=lambda: _env_int("FEDMED_MAX_EPOCHS", 100))

    @property
    def device(self) -> torch.device:
        """CUDA if available, otherwise CPU. Read this everywhere instead of hardcoding a device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Shared default instance. Import this directly for scripts that don't need custom
# overrides; construct a fresh `BraTSConfig(...)` when a script needs different values.
DEFAULT_CONFIG = BraTSConfig()
