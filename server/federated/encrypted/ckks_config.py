"""CKKS encryption configuration -- the single documented source of encryption
parameters. `FEDMED_CKKS_*`-env-overridable, same dataclass pattern as every other
config in this project; nothing hard-coded at its point of use.

Defaults (`poly_modulus_degree=8192`, `coeff_mod_bit_sizes=[60, 40, 40, 60]`,
`global_scale=2**40`) are a standard, documented CKKS development configuration -- 128-bit
security range at this poly_modulus_degree/coeff-modulus combination, two usable
multiplicative levels (only one is actually used: a single plaintext-scalar multiply per
chunk for FedAvg weighting -- see server/federated/encrypted/encryption.py). Not chosen
at random.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(var_name: str, default: int) -> int:
    return int(os.environ.get(var_name, default))


def _env_float(var_name: str, default: float) -> float:
    return float(os.environ.get(var_name, default))


def _env_int_list(var_name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.environ.get(var_name)
    if raw is None:
        return default
    return tuple(int(part) for part in raw.split(","))


@dataclass(frozen=True)
class CKKSConfig:
    """Immutable CKKS parameters + the operational settings that depend on them."""

    poly_modulus_degree: int = field(default_factory=lambda: _env_int("FEDMED_CKKS_POLY_MODULUS_DEGREE", 8192))
    coeff_mod_bit_sizes: tuple[int, ...] = field(
        default_factory=lambda: _env_int_list("FEDMED_CKKS_COEFF_MOD_BIT_SIZES", (60, 40, 40, 60))
    )
    global_scale: float = field(default_factory=lambda: _env_float("FEDMED_CKKS_GLOBAL_SCALE", 2.0**40))

    # CKKS slot capacity is poly_modulus_degree / 2 -- a value cannot safely encrypt more
    # elements than this in one CKKSVector (TenSEAL silently falls back to a
    # matmul-disabling auto-batching mode above that, which this project deliberately
    # does not rely on -- see docs/homomorphic_encryption.md). Defaults to the full slot
    # capacity; overridable smaller (e.g. for fast tests that want multiple chunks).
    chunk_size: int = field(default_factory=lambda: _env_int("FEDMED_CKKS_CHUNK_SIZE", 0))

    # Set from this module's own measured numerical-accuracy test
    # (server/tests/test_ckks_aggregation.py), not guessed.
    numerical_tolerance: float = field(default_factory=lambda: _env_float("FEDMED_CKKS_NUMERICAL_TOLERANCE", 1e-3))

    def __post_init__(self) -> None:
        if self.chunk_size == 0:
            object.__setattr__(self, "chunk_size", self.poly_modulus_degree // 2)

    @property
    def slot_capacity(self) -> int:
        return self.poly_modulus_degree // 2


DEFAULT_CONFIG = CKKSConfig()
