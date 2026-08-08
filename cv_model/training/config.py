"""Training-loop and model-architecture configuration for the centralized baseline.

Deliberately separate from `cv_model.brats.config.BraTSRawConfig` (Module 3):
that config owns everything about the *data* (paths, batch size, patch size,
channel/class counts derived from the dataset); this one owns everything
about *how a model is built and trained* on that data. `train_baseline`
(`trainer.py`) takes one of each rather than a single merged config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch


def _env_path(var_name: str, default: str) -> Path:
    return Path(os.environ.get(var_name, default)).expanduser().resolve()


def _env_str(var_name: str, default: str) -> str:
    return os.environ.get(var_name, default)


def _env_int(var_name: str, default: int) -> int:
    return int(os.environ.get(var_name, default))


def _env_float(var_name: str, default: float) -> float:
    return float(os.environ.get(var_name, default))


def _env_bool(var_name: str, default: bool) -> bool:
    raw = os.environ.get(var_name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_int(var_name: str) -> int | None:
    raw = os.environ.get(var_name)
    return None if raw is None else int(raw)


def _env_optional_float(var_name: str) -> float | None:
    raw = os.environ.get(var_name)
    return None if raw is None else float(raw)


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable configuration for the centralized training baseline."""

    # --- Reproducibility ---
    # Separate from BraTSRawConfig.seed (which only controls the train/val split):
    # this seeds torch/numpy/random for weight init and augmentation order.
    seed: int = field(default_factory=lambda: _env_int("FEDMED_TRAIN_SEED", 42))

    # --- Model architecture (kept here, not in BraTSRawConfig, since these are
    # training/architecture decisions independent of how the data was loaded) ---
    unet_channels: tuple[int, ...] = (16, 32, 64, 128, 256)
    unet_strides: tuple[int, ...] = (2, 2, 2, 2)
    unet_num_res_units: int = 2

    # --- Optimization ---
    epochs: int = field(default_factory=lambda: _env_int("FEDMED_TRAIN_EPOCHS", 50))
    learning_rate: float = field(default_factory=lambda: _env_float("FEDMED_TRAIN_LR", 1e-4))
    optimizer_name: Literal["adam", "sgd"] = field(
        default_factory=lambda: _env_str("FEDMED_TRAIN_OPTIMIZER", "adam")  # type: ignore[return-value]
    )
    weight_decay: float = field(default_factory=lambda: _env_float("FEDMED_TRAIN_WEIGHT_DECAY", 1e-5))
    grad_clip_norm: float | None = field(
        default_factory=lambda: _env_optional_float("FEDMED_TRAIN_GRAD_CLIP_NORM")  # None unless set
    )

    # --- Scheduler (optional) ---
    scheduler_type: Literal["none", "step", "cosine"] = field(
        default_factory=lambda: _env_str("FEDMED_TRAIN_SCHEDULER", "none")  # type: ignore[return-value]
    )
    scheduler_step_size: int = field(default_factory=lambda: _env_int("FEDMED_TRAIN_SCHEDULER_STEP_SIZE", 20))
    scheduler_gamma: float = field(default_factory=lambda: _env_float("FEDMED_TRAIN_SCHEDULER_GAMMA", 0.5))

    # --- Device / precision ---
    # "auto" picks CUDA if available, else CPU -- never assumes a GPU exists.
    device_preference: Literal["auto", "cpu", "cuda"] = field(
        default_factory=lambda: _env_str("FEDMED_TRAIN_DEVICE", "auto")  # type: ignore[return-value]
    )
    mixed_precision: bool = field(default_factory=lambda: _env_bool("FEDMED_TRAIN_MIXED_PRECISION", False))

    # --- Validation / early stopping / checkpointing ---
    # Run validation every N epochs (1 = every epoch).
    val_frequency: int = field(default_factory=lambda: _env_int("FEDMED_TRAIN_VAL_FREQUENCY", 1))
    # None disables early stopping.
    early_stopping_patience: int | None = field(
        default_factory=lambda: _env_optional_int("FEDMED_TRAIN_EARLY_STOPPING_PATIENCE")
    )
    checkpoint_dir: Path = field(
        default_factory=lambda: _env_path("FEDMED_TRAIN_CHECKPOINT_DIR", "./checkpoints/brats_baseline")
    )
    resume_from: Path | None = None

    @property
    def device(self) -> torch.device:
        if self.device_preference == "cpu":
            return torch.device("cpu")
        if self.device_preference == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "FEDMED_TRAIN_DEVICE=cuda was requested but torch.cuda.is_available() is False. "
                    "Install a CUDA-enabled PyTorch build, or set FEDMED_TRAIN_DEVICE=auto/cpu."
                )
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEFAULT_CONFIG = TrainingConfig()
