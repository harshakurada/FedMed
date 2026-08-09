from __future__ import annotations

import numpy as np
import pytest
import tenseal as ts
from flwr.server.strategy.aggregate import aggregate as fedavg_aggregate

from server.federated.encrypted.ckks_config import CKKSConfig
from server.federated.encrypted.encryption import encrypt_model_update, homomorphically_add_updates
from server.federated.encrypted.key_holder import KeyHolder
from server.federated.encrypted.serialization import ParamSpec


@pytest.fixture
def ckks_config() -> CKKSConfig:
    return CKKSConfig(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60), global_scale=2.0**40, chunk_size=4)


def _make_update(key_holder: KeyHolder, ckks_config: CKKSConfig, values, hospital_id: str, num_examples: int):
    array = np.array(values, dtype=np.float64)
    specs = [ParamSpec(name="x", shape=(len(values),), dtype="float32", offset=0, length=len(values))]
    update = encrypt_model_update(
        array, specs, key_holder.public_context_bytes, ckks_config.chunk_size, hospital_id, 1, "v1", num_examples
    )
    return update, specs


def test_homomorphic_addition_of_two_known_vectors(ckks_config: CKKSConfig) -> None:
    key_holder = KeyHolder.generate(ckks_config)
    update_a, specs = _make_update(key_holder, ckks_config, [1.0, 2.0, 3.0], "hospital_a", num_examples=1)
    update_b, _ = _make_update(key_holder, ckks_config, [10.0, 20.0, 30.0], "hospital_b", num_examples=1)

    aggregated_chunks = homomorphically_add_updates([update_a, update_b], key_holder.public_context_bytes)
    recovered = key_holder.decrypt_aggregate(aggregated_chunks, specs, total_examples=2)
    assert recovered["x"].numpy() == pytest.approx([5.5, 11.0, 16.5], abs=1e-3)


def test_weighted_aggregation_matches_plaintext_fedavg_within_tolerance(ckks_config: CKKSConfig) -> None:
    """The numerical-accuracy test: same 3 client updates, aggregated both ways,
    compared against Flower's own real FedAvg weighting function."""
    key_holder = KeyHolder.generate(ckks_config)
    rng = np.random.default_rng(0)
    values_a = rng.uniform(-1, 1, size=20)
    values_b = rng.uniform(-1, 1, size=20)
    values_c = rng.uniform(-1, 1, size=20)
    n_a, n_b, n_c = 10, 30, 60

    update_a, specs = _make_update(key_holder, ckks_config, values_a, "hospital_a", n_a)
    update_b, _ = _make_update(key_holder, ckks_config, values_b, "hospital_b", n_b)
    update_c, _ = _make_update(key_holder, ckks_config, values_c, "hospital_c", n_c)

    aggregated_chunks = homomorphically_add_updates([update_a, update_b, update_c], key_holder.public_context_bytes)
    encrypted_result = key_holder.decrypt_aggregate(aggregated_chunks, specs, total_examples=n_a + n_b + n_c)[
        "x"
    ].numpy()

    plaintext_result = fedavg_aggregate([([values_a], n_a), ([values_b], n_b), ([values_c], n_c)])[0]

    abs_errors = np.abs(plaintext_result - encrypted_result)
    max_abs_error = float(abs_errors.max())
    mean_abs_error = float(abs_errors.mean())
    relative_error = float(np.mean(abs_errors / np.maximum(np.abs(plaintext_result), 1e-12)))
    print(
        f"max_abs_error={max_abs_error:.3e} mean_abs_error={mean_abs_error:.3e} relative_error={relative_error:.3e}"
    )
    assert max_abs_error < ckks_config.numerical_tolerance


def test_homomorphic_aggregation_never_decrypts_an_individual_update(ckks_config: CKKSConfig, monkeypatch) -> None:
    """Security test: aggregation must operate on ciphertexts only. If this ever calls
    .decrypt() on an individual update, that's a bug that must be fixed, not tolerated."""
    key_holder = KeyHolder.generate(ckks_config)
    update_a, _specs = _make_update(key_holder, ckks_config, [1.0, 2.0], "hospital_a", 5)
    update_b, _ = _make_update(key_holder, ckks_config, [3.0, 4.0], "hospital_b", 7)

    decrypt_calls: list[bool] = []
    original_decrypt = ts.CKKSVector.decrypt

    def spy_decrypt(self, *args, **kwargs):
        decrypt_calls.append(True)
        return original_decrypt(self, *args, **kwargs)

    monkeypatch.setattr(ts.CKKSVector, "decrypt", spy_decrypt)
    homomorphically_add_updates([update_a, update_b], key_holder.public_context_bytes)

    assert decrypt_calls == [], "homomorphically_add_updates must never call .decrypt() on an individual update"


def test_mismatched_chunk_counts_are_rejected(ckks_config: CKKSConfig) -> None:
    key_holder = KeyHolder.generate(ckks_config)
    update_a, _ = _make_update(key_holder, ckks_config, [1.0, 2.0, 3.0, 4.0, 5.0], "hospital_a", 1)  # 2 chunks
    update_b, _ = _make_update(key_holder, ckks_config, [1.0, 2.0], "hospital_b", 1)  # 1 chunk
    with pytest.raises(ValueError, match="same number of chunks"):
        homomorphically_add_updates([update_a, update_b], key_holder.public_context_bytes)
