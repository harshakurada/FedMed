"""The authorized decrypting component -- the ONLY place in this project that ever
holds a CKKS private context or calls `.decrypt()`.

Structurally distinct from both the hospitals and the aggregation server
(`server/federated/encrypted/aggregator.py`): neither of those modules imports this one
or constructs a private context anywhere in their own code (verified in
`server/tests/test_ckks_security.py`, not just documented). Models "a party the
aggregation server is not" for this local simulation -- see
docs/homomorphic_encryption.md for the full threat model and key-ownership table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tenseal as ts
import torch

from server.federated.encrypted.ckks_config import CKKSConfig
from server.federated.encrypted.serialization import ParamSpec, unflatten_model_parameters


@dataclass
class KeyHolder:
    """Owns the CKKS private context. `public_context_bytes` is the only thing ever
    handed to a hospital or the aggregation server -- serialized with
    `save_secret_key=False`, and (since this project only ever needs ciphertext-
    ciphertext addition and plaintext-scalar multiplication for FedAvg weighting, never
    ciphertext-ciphertext multiplication or slot rotation) `save_galois_keys=False` too,
    measured ~75x smaller in this project's own testing (35 MB -> ~465 KB)."""

    _context: ts.Context

    @classmethod
    def generate(cls, config: CKKSConfig) -> "KeyHolder":
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=config.poly_modulus_degree,
            coeff_mod_bit_sizes=list(config.coeff_mod_bit_sizes),
        )
        context.global_scale = config.global_scale
        return cls(_context=context)

    @property
    def public_context_bytes(self) -> bytes:
        return self._context.serialize(save_secret_key=False, save_galois_keys=False, save_relin_keys=False)

    def decrypt_aggregate(
        self, chunk_ciphertexts: list[bytes], param_specs: list[ParamSpec], total_examples: int
    ) -> dict[str, torch.Tensor]:
        """The only place `.decrypt()` is called anywhere in this project. Divides the
        homomorphically-weighted sum by `total_examples` here, once, in plaintext --
        mathematically identical to normalizing before encryption, but avoids an extra
        homomorphic multiply (see `encryption.py`'s module docstring)."""
        if total_examples <= 0:
            raise ValueError(f"total_examples must be > 0, got {total_examples}")

        values: list[float] = []
        for chunk_bytes in chunk_ciphertexts:
            vector = ts.lazy_ckks_vector_from(chunk_bytes)
            vector.link_context(self._context)
            values.extend(vector.decrypt())

        weighted_sum = np.array(values, dtype=np.float64)
        averaged = weighted_sum / total_examples
        return unflatten_model_parameters(averaged, param_specs)
