"""Planned TenSEAL configuration (Week 3+).

Only the parameters a homomorphic-encryption context will eventually need
live here -- no keys, no context, no ciphertext logic. Keeps every future
encrypt/decrypt call reading scheme parameters from one place instead of
scattering them across the server/client code that uses them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_str(var_name: str, default: str) -> str:
    return os.environ.get(var_name, default)


def _env_int(var_name: str, default: int) -> int:
    return int(os.environ.get(var_name, default))


@dataclass(frozen=True)
class EncryptionConfig:
    # CKKS supports encrypted floating-point arithmetic, needed for weight
    # aggregation on ciphertext (Week 3's homomorphic FedAvg).
    scheme: str = field(default_factory=lambda: _env_str("FEDMED_HE_SCHEME", "CKKS"))
    poly_modulus_degree: int = field(
        default_factory=lambda: _env_int("FEDMED_HE_POLY_MODULUS_DEGREE", 8192)
    )
    global_scale_bits: int = field(default_factory=lambda: _env_int("FEDMED_HE_GLOBAL_SCALE_BITS", 40))


DEFAULT_CONFIG = EncryptionConfig()
