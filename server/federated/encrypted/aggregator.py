"""The aggregation server: holds ONLY a public CKKS context (loaded from bytes
containing no secret key). There is no code path anywhere in this class that constructs
a private context or calls `.decrypt()` -- this is what makes "the server cannot decrypt
individual updates" a structural fact, not just a policy (verified in
`server/tests/test_ckks_security.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

from server.federated.encrypted.encryption import EncryptedUpdate, homomorphically_add_updates
from server.federated.encrypted.fingerprint import compute_context_fingerprint
from server.federated.encrypted.serialization import ModelMetadataError, ParamSpec, validate_model_metadata


@dataclass
class EncryptedAggregate:
    round_id: int
    model_version: str
    total_examples: int
    param_specs: list[ParamSpec]
    chunk_ciphertexts: list[bytes]


class EncryptedAggregationServer:
    """Never constructs a private CKKS context. `public_context_bytes` is the only
    cryptographic material this class ever holds."""

    def __init__(
        self,
        public_context_bytes: bytes,
        expected_param_specs: list[ParamSpec],
        round_id: int,
        model_version: str,
    ) -> None:
        self.public_context_bytes = public_context_bytes
        self._expected_context_fingerprint = compute_context_fingerprint(public_context_bytes)
        self._expected_param_specs = expected_param_specs
        self._round_id = round_id
        self._model_version = model_version
        self._updates: list[EncryptedUpdate] = []
        self._seen_hospital_ids: set[str] = set()

    def submit_update(self, update: EncryptedUpdate) -> None:
        # CKKS itself doesn't fail closed on a wrong context (see fingerprint.py) --
        # this check is what actually makes "wrong context is rejected" true.
        if update.context_fingerprint != self._expected_context_fingerprint:
            raise ModelMetadataError(
                f"context fingerprint mismatch for {update.hospital_id}: this update was "
                "encrypted under a different CKKS context than the one this aggregation "
                "server expects -- rejected before aggregation."
            )
        metadata = {
            "round_id": update.round_id,
            "model_version": update.model_version,
            "hospital_id": update.hospital_id,
            "num_examples": update.num_examples,
            "param_specs": update.param_specs,
        }
        validate_model_metadata(metadata, self._expected_param_specs, self._round_id, self._model_version)
        if update.hospital_id in self._seen_hospital_ids:
            # Replay protection: a second update claiming the same hospital+round is
            # rejected rather than silently double-counted in the aggregate.
            raise ValueError(f"Duplicate update from {update.hospital_id} for round {update.round_id} -- rejected.")
        self._seen_hospital_ids.add(update.hospital_id)
        self._updates.append(update)

    def aggregate(self) -> EncryptedAggregate:
        if not self._updates:
            raise RuntimeError("No updates submitted -- nothing to aggregate.")
        aggregated_chunks = homomorphically_add_updates(self._updates, self.public_context_bytes)
        total_examples = sum(update.num_examples for update in self._updates)
        return EncryptedAggregate(
            round_id=self._round_id,
            model_version=self._model_version,
            total_examples=total_examples,
            param_specs=self._expected_param_specs,
            chunk_ciphertexts=aggregated_chunks,
        )
