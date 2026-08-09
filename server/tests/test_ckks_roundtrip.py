from __future__ import annotations

import numpy as np
import pytest
import tenseal as ts

from server.federated.encrypted.chunking import chunk_values, unchunk_values
from server.federated.encrypted.ckks_config import CKKSConfig
from server.federated.encrypted.key_holder import KeyHolder
from server.federated.encrypted.serialization import (
    ParamSpec,
    deserialize_model_metadata,
    serialize_model_metadata,
)


@pytest.fixture
def ckks_config() -> CKKSConfig:
    return CKKSConfig(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60), global_scale=2.0**40, chunk_size=4)


def test_encrypt_decrypt_round_trip_small_vector_within_tolerance(ckks_config: CKKSConfig) -> None:
    key_holder = KeyHolder.generate(ckks_config)
    context = ts.context_from(key_holder.public_context_bytes)
    values = [1.0, -2.5, 3.75, 0.0]
    ciphertext = ts.ckks_vector(context, values).serialize()

    recovered = key_holder.decrypt_aggregate(
        [ciphertext], [ParamSpec("x", (4,), "float32", 0, 4)], total_examples=1
    )
    errors = np.abs(np.array(values) - recovered["x"].numpy())
    print(f"round-trip max_abs_error={errors.max():.3e}")
    assert errors.max() < ckks_config.numerical_tolerance


def test_chunking_splits_and_reconstructs_values_larger_than_one_chunk() -> None:
    values = np.arange(10, dtype=np.float64)
    chunks = chunk_values(values, chunk_size=4)
    assert [len(c) for c in chunks] == [4, 4, 2]
    assert np.array_equal(unchunk_values(chunks), values)


def test_chunking_exact_multiple_of_chunk_size() -> None:
    values = np.arange(8, dtype=np.float64)
    chunks = chunk_values(values, chunk_size=4)
    assert [len(c) for c in chunks] == [4, 4]


def test_metadata_round_trips_through_json_serialization() -> None:
    specs = [ParamSpec(name="conv1.weight", shape=(2, 3), dtype="float32", offset=0, length=6)]
    data = serialize_model_metadata(specs, round_id=3, model_version="v2", hospital_id="hospital_a", num_examples=42)
    restored = deserialize_model_metadata(data)
    assert restored["round_id"] == 3
    assert restored["model_version"] == "v2"
    assert restored["hospital_id"] == "hospital_a"
    assert restored["num_examples"] == 42
    assert restored["param_specs"] == specs


def test_ciphertext_serialize_deserialize_then_homomorphic_op(ckks_config: CKKSConfig) -> None:
    key_holder = KeyHolder.generate(ckks_config)
    context = ts.context_from(key_holder.public_context_bytes)
    bytes1 = ts.ckks_vector(context, [1.0, 2.0]).serialize()
    bytes2 = ts.ckks_vector(context, [10.0, 20.0]).serialize()

    # Deserialize under a freshly-loaded (but same) public context -- proves the
    # ciphertext transport representation survives a real serialize/deserialize cycle.
    reloaded_context = ts.context_from(key_holder.public_context_bytes)
    r1 = ts.lazy_ckks_vector_from(bytes1)
    r1.link_context(reloaded_context)
    r2 = ts.lazy_ckks_vector_from(bytes2)
    r2.link_context(reloaded_context)
    total_bytes = (r1 + r2).serialize()

    recovered = key_holder.decrypt_aggregate([total_bytes], [ParamSpec("x", (2,), "float32", 0, 2)], total_examples=1)
    assert recovered["x"].numpy() == pytest.approx([11.0, 22.0], abs=1e-3)
