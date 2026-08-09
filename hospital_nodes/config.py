"""Per-hospital configuration for the 3 mock hospital nodes (Hospital A/B/C).

Each hospital is an independent Flower SuperNode: its own client identity,
its own local data directory, and its own ClientAppIo port. No hospital's
config ever points at another hospital's data directory -- mirrors the real
constraint that raw patient data never leaves a hospital.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from cv_model.training.config import TrainingConfig


def _env_str(var_name: str, default: str) -> str:
    return os.environ.get(var_name, default)


def _env_int(var_name: str, default: int) -> int:
    return int(os.environ.get(var_name, default))


def _env_float(var_name: str, default: float) -> float:
    return float(os.environ.get(var_name, default))


def _env_path(var_name: str, default: str) -> Path:
    return Path(os.environ.get(var_name, default)).expanduser().resolve()


@dataclass(frozen=True)
class HospitalConfig:
    """Identity and local settings for a single mock hospital node."""

    partition_id: int
    name: str
    data_root: Path
    clientappio_port: int


def _hospital(partition_id: int, default_name: str, default_port: int) -> HospitalConfig:
    prefix = f"FEDMED_HOSPITAL_{partition_id}"
    return HospitalConfig(
        partition_id=partition_id,
        name=_env_str(f"{prefix}_NAME", default_name),
        data_root=Path(_env_str(f"{prefix}_DATA_ROOT", f"./data/hospital_{partition_id}")).expanduser().resolve(),
        clientappio_port=_env_int(f"{prefix}_PORT", default_port),
    )


# The 3 mock hospitals this project simulates -- each gets its own SuperNode
# process, its own local data directory, and its own ClientAppIo port.
HOSPITAL_NODES: tuple[HospitalConfig, ...] = (
    _hospital(0, "Hospital A", 9094),
    _hospital(1, "Hospital B", 9095),
    _hospital(2, "Hospital C", 9096),
)


def get_hospital_config(partition_id: int) -> HospitalConfig:
    """Look up a hospital's config by its partition id (0, 1, or 2)."""
    for hospital in HOSPITAL_NODES:
        if hospital.partition_id == partition_id:
            return hospital
    raise ValueError(f"No hospital configured for partition_id={partition_id}")


# Root directory each hospital's checkpoints live under, one subfolder per hospital
# (e.g. checkpoints/hospitals/hospital_a/) -- never a shared path, so one hospital can
# never overwrite another's. Rooted under the same gitignored `checkpoints/` convention
# Module 4/5 already established, not a second parallel output tree.
HOSPITAL_CHECKPOINT_ROOT: Path = _env_path("FEDMED_HOSPITAL_CHECKPOINT_ROOT", "./checkpoints/hospitals")


@dataclass(frozen=True)
class HospitalTrainingConfig:
    """Local-training settings for one hospital: composes the existing shared
    `TrainingConfig` (Module 4/5 -- model architecture, optimizer, device, ...) with only
    the hospital-specific overrides, rather than duplicating its fields."""

    hospital: HospitalConfig
    train_config: TrainingConfig
    # Epochs per local training call (e.g. one federated round) -- deliberately a
    # separate concept from TrainingConfig.epochs (a full centralized run's epoch count).
    local_epochs: int
    # 0.0 (default): the hospital's entire partition trains locally, no local validation
    # (Module 6: "if the centralized validation set is already defined as a global
    # holdout, preserve it" -- evaluation happens against that shared holdout instead).
    # >0.0: carve a further deterministic patient-level split *within this hospital's own
    # partition only* -- never touches another hospital's data or the global val set.
    local_val_fraction: float


def build_hospital_training_config(
    partition_id: int,
    base_train_config: TrainingConfig | None = None,
    local_epochs: int | None = None,
    local_val_fraction: float | None = None,
) -> HospitalTrainingConfig:
    """Derive one hospital's training config from the shared base, overriding only
    `checkpoint_dir` (made unique per hospital) and the hospital-specific fields."""
    hospital = get_hospital_config(partition_id)
    base_train_config = base_train_config or TrainingConfig()
    checkpoint_dir = HOSPITAL_CHECKPOINT_ROOT / f"hospital_{chr(ord('a') + partition_id)}"
    train_config = replace(base_train_config, checkpoint_dir=checkpoint_dir)
    return HospitalTrainingConfig(
        hospital=hospital,
        train_config=train_config,
        local_epochs=local_epochs if local_epochs is not None else _env_int("FEDMED_HOSPITAL_LOCAL_EPOCHS", 1),
        local_val_fraction=(
            local_val_fraction
            if local_val_fraction is not None
            else _env_float("FEDMED_HOSPITAL_LOCAL_VAL_FRACTION", 0.0)
        ),
    )
