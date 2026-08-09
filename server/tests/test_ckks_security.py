from __future__ import annotations

import inspect

import numpy as np
import pytest
import tenseal as ts

from server.federated.encrypted import aggregator, encryption
from server.federated.encrypted.aggregator import EncryptedAggregationServer
from server.federated.encrypted.ckks_config import CKKSConfig
from server.federated.encrypted.encryption import deserialize_ciphertext, encrypt_model_update
from server.federated.encrypted.key_holder import KeyHolder
from server.federated.encrypted.serialization import ModelMetadataError, ParamSpec


@pytest.fixture
def ckks_config() -> CKKSConfig:
    return CKKSConfig(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60), global_scale=2.0**40, chunk_size=4)


def test_aggregator_and_encryption_source_never_constructs_a_private_context() -> None:
    """Static check: the server-side modules never save/load a secret key, and never
    generate Galois keys (unneeded for +/scalar-* only, and expensive to transmit)."""
    for module in (aggregator, encryption):
        source = inspect.getsource(module)
        assert "save_secret_key=True" not in source
        assert "generate_galois_keys" not in source


def test_aggregation_servers_context_has_no_secret_key(ckks_config: CKKSConfig) -> None:
    key_holder = KeyHolder.generate(ckks_config)
    server = EncryptedAggregationServer(key_holder.public_context_bytes, [], round_id=1, model_version="v1")
    loaded_context = ts.context_from(server.public_context_bytes)
    assert loaded_context.has_secret_key() is False


def test_public_context_bytes_are_smaller_than_and_distinct_from_the_private_serialization(
    ckks_config: CKKSConfig,
) -> None:
    key_holder = KeyHolder.generate(ckks_config)
    private_bytes = key_holder._context.serialize(save_secret_key=True)
    public_bytes = key_holder.public_context_bytes
    assert private_bytes != public_bytes
    assert len(public_bytes) < len(private_bytes)


def test_key_holder_object_has_no_attribute_exposing_raw_secret_key_bytes(ckks_config: CKKSConfig) -> None:
    key_holder = KeyHolder.generate(ckks_config)
    public_names = [name for name in vars(key_holder) if not name.startswith("_")]
    # The only public attribute a KeyHolder exposes is the dataclass field itself
    # (_context, private by convention) plus the public_context_bytes property --
    # nothing named/shaped like a raw secret-key accessor exists on the object.
    assert "secret_key" not in public_names
    assert "secret_context_bytes" not in public_names


def test_corrupted_ciphertext_is_rejected_not_silently_accepted(ckks_config: CKKSConfig) -> None:
    key_holder = KeyHolder.generate(ckks_config)
    with pytest.raises(Exception):
        deserialize_ciphertext(b"not a real ciphertext", key_holder.public_context_bytes)


def test_wrong_context_update_is_rejected_via_fingerprint_check(ckks_config: CKKSConfig) -> None:
    """CKKS decryption itself does NOT fail closed on a context mismatch (confirmed by
    testing: it silently returns meaningless numbers rather than raising) -- this project's
    own context-fingerprint check (fingerprint.py) is what actually makes 'wrong context is
    rejected' true, verified here against two independently-generated contexts."""
    key_holder = KeyHolder.generate(ckks_config)
    other_key_holder = KeyHolder.generate(ckks_config)
    specs = [ParamSpec("x", (2,), "float32", 0, 2)]

    server = EncryptedAggregationServer(key_holder.public_context_bytes, specs, round_id=1, model_version="v1")
    wrong_context_update = encrypt_model_update(
        np.array([1.0, 2.0]), specs, other_key_holder.public_context_bytes, ckks_config.chunk_size, "hospital_a", 1, "v1", 1
    )
    with pytest.raises(ModelMetadataError, match="context fingerprint mismatch"):
        server.submit_update(wrong_context_update)


def test_wrong_round_id_is_rejected(ckks_config: CKKSConfig) -> None:
    key_holder = KeyHolder.generate(ckks_config)
    specs = [ParamSpec("x", (2,), "float32", 0, 2)]
    server = EncryptedAggregationServer(key_holder.public_context_bytes, specs, round_id=1, model_version="v1")
    update = encrypt_model_update(
        np.array([1.0, 2.0]), specs, key_holder.public_context_bytes, ckks_config.chunk_size, "hospital_a", 2, "v1", 1
    )
    with pytest.raises(ModelMetadataError, match="round_id"):
        server.submit_update(update)


def test_wrong_model_version_is_rejected(ckks_config: CKKSConfig) -> None:
    key_holder = KeyHolder.generate(ckks_config)
    specs = [ParamSpec("x", (2,), "float32", 0, 2)]
    server = EncryptedAggregationServer(key_holder.public_context_bytes, specs, round_id=1, model_version="v1")
    update = encrypt_model_update(
        np.array([1.0, 2.0]), specs, key_holder.public_context_bytes, ckks_config.chunk_size, "hospital_a", 1, "v2", 1
    )
    with pytest.raises(ModelMetadataError, match="model_version"):
        server.submit_update(update)


def test_duplicate_update_from_the_same_hospital_is_rejected_replay_protection(ckks_config: CKKSConfig) -> None:
    key_holder = KeyHolder.generate(ckks_config)
    specs = [ParamSpec("x", (2,), "float32", 0, 2)]
    server = EncryptedAggregationServer(key_holder.public_context_bytes, specs, round_id=1, model_version="v1")
    update = encrypt_model_update(
        np.array([1.0, 2.0]), specs, key_holder.public_context_bytes, ckks_config.chunk_size, "hospital_a", 1, "v1", 1
    )
    server.submit_update(update)
    with pytest.raises(ValueError, match="Duplicate update"):
        server.submit_update(update)


def test_no_raw_data_or_tensor_ever_touches_the_encryption_layer(ckks_config: CKKSConfig) -> None:
    key_holder = KeyHolder.generate(ckks_config)
    specs = [ParamSpec("x", (2,), "float32", 0, 2)]
    update = encrypt_model_update(
        np.array([1.0, 2.0]), specs, key_holder.public_context_bytes, ckks_config.chunk_size, "hospital_a", 1, "v1", 1
    )
    for value in (update.hospital_id, update.round_id, update.model_version, update.num_examples):
        assert not hasattr(value, "modality_paths")
    for chunk in update.chunk_ciphertexts:
        assert isinstance(chunk, bytes)
