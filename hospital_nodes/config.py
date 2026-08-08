"""Per-hospital configuration for the 3 mock hospital nodes (Hospital A/B/C).

Each hospital is an independent Flower SuperNode: its own client identity,
its own local data directory, and its own ClientAppIo port. No hospital's
config ever points at another hospital's data directory -- mirrors the real
constraint that raw patient data never leaves a hospital.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_str(var_name: str, default: str) -> str:
    return os.environ.get(var_name, default)


def _env_int(var_name: str, default: int) -> int:
    return int(os.environ.get(var_name, default))


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
