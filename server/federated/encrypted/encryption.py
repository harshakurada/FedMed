"""CKKS encryption utilities: per-hospital encryption of a flattened model update, and
the server-side homomorphic weighted-sum aggregation.

**Weighting**: clients encrypt their raw, unweighted flattened update. The server
homomorphically computes `sum_i(num_examples_i * ciphertext_i)` -- plaintext-scalar
multiplication (a public, non-sensitive integer) then ciphertext-ciphertext addition,
neither of which needs Galois or relinearization keys. The final division by
`total_examples` happens once, in plaintext, in `KeyHolder.decrypt_aggregate` -- dividing
a decrypted value by a known public scalar is exact and free, and this mirrors exactly
how `flwr.server.strategy.aggregate.aggregate` structures plaintext FedAvg (weight-then-
sum, normalize once at the end), making the two directly comparable.

`homomorphically_add_updates` never calls `.decrypt()` -- verified in
`server/tests/test_ckks_aggregation.py`, which spies on `CKKSVector.decrypt` during this
function and asserts it is never invoked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tenseal as ts

from server.federated.encrypted.chunking import chunk_values
from server.federated.encrypted.fingerprint import compute_context_fingerprint
from server.federated.encrypted.serialization import ParamSpec


@dataclass
class EncryptedUpdate:
    """One hospital's encrypted contribution to one round. Only ciphertext bytes and
    approved metadata -- never raw data, never a plaintext parameter value.
    `context_fingerprint` lets the aggregation server reject an update encrypted under a
    different context than the one it expects (see `fingerprint.py` for why this is
    needed -- CKKS itself doesn't fail closed on a context mismatch)."""

    hospital_id: str
    round_id: int
    model_version: str
    num_examples: int
    param_specs: list[ParamSpec]
    chunk_ciphertexts: list[bytes]
    context_fingerprint: str


def encrypt_model_update(
    flattened: np.ndarray,
    param_specs: list[ParamSpec],
    public_context_bytes: bytes,
    chunk_size: int,
    hospital_id: str,
    round_id: int,
    model_version: str,
    num_examples: int,
) -> EncryptedUpdate:
    """Encrypts a hospital's own flattened update using only a public context, loaded
    fresh from bytes -- never a live reference to a `KeyHolder`'s private context."""
    context = ts.context_from(public_context_bytes)
    chunks = chunk_values(flattened, chunk_size)
    chunk_ciphertexts = [ts.ckks_vector(context, chunk.tolist()).serialize() for chunk in chunks]
    return EncryptedUpdate(
        hospital_id=hospital_id,
        round_id=round_id,
        model_version=model_version,
        num_examples=num_examples,
        param_specs=param_specs,
        chunk_ciphertexts=chunk_ciphertexts,
        context_fingerprint=compute_context_fingerprint(public_context_bytes),
    )


def homomorphically_add_updates(updates: list[EncryptedUpdate], public_context_bytes: bytes) -> list[bytes]:
    """The server-side homomorphic weighted sum. Operates on `CKKSVector` objects only
    -- no update is ever individually decrypted here (see module docstring)."""
    if not updates:
        raise ValueError("homomorphically_add_updates requires at least one update.")

    num_chunks = len(updates[0].chunk_ciphertexts)
    if any(len(update.chunk_ciphertexts) != num_chunks for update in updates):
        raise ValueError("All updates must have the same number of chunks to aggregate.")

    context = ts.context_from(public_context_bytes)
    aggregated_chunks: list[bytes] = []
    for chunk_index in range(num_chunks):
        weighted_sum = None
        for update in updates:
            vector = ts.lazy_ckks_vector_from(update.chunk_ciphertexts[chunk_index])
            vector.link_context(context)
            weighted = vector * update.num_examples  # plaintext-scalar multiply, no relin/galois keys needed
            weighted_sum = weighted if weighted_sum is None else weighted_sum + weighted
        aggregated_chunks.append(weighted_sum.serialize())
    return aggregated_chunks


def serialize_ciphertext(vector: "ts.CKKSVector") -> bytes:
    return vector.serialize()


def deserialize_ciphertext(data: bytes, public_context_bytes: bytes) -> "ts.CKKSVector":
    context = ts.context_from(public_context_bytes)
    vector = ts.lazy_ckks_vector_from(data)
    vector.link_context(context)
    return vector
