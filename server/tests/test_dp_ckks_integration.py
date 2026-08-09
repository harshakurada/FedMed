"""DP update -> CKKS encryption -> homomorphic aggregation -> authorized decryption,
compared against the corresponding plaintext DP aggregation. The SAME DP-protected
values are used for both branches (computed once, upstream of the plaintext-vs-encrypted
fork) -- this isolates CKKS's own numerical error from DP's independent stochasticity,
exactly as the module plan requires.
"""

from __future__ import annotations

import numpy as np
import pytest
from flwr.server.strategy.aggregate import aggregate as fedavg_aggregate

from server.federated.dp.dp_config import DPConfig
from server.federated.dp.dp_update import apply_dp_mechanism
from server.federated.encrypted.aggregator import EncryptedAggregationServer
from server.federated.encrypted.ckks_config import CKKSConfig
from server.federated.encrypted.encryption import encrypt_model_update
from server.federated.encrypted.key_holder import KeyHolder
from server.federated.encrypted.serialization import ParamSpec


@pytest.fixture
def ckks_config() -> CKKSConfig:
    return CKKSConfig(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60), global_scale=2.0**40, chunk_size=8)


def test_dp_protected_updates_aggregate_correctly_through_ckks(ckks_config: CKKSConfig) -> None:
    dp_config = DPConfig(enabled=True, clip_norm=1.0, noise_multiplier=3.0, delta=1e-5)

    raw_updates = [
        ("hospital_a", np.zeros(10), np.random.default_rng(100).uniform(-0.5, 0.5, 10), 5),
        ("hospital_b", np.zeros(10), np.random.default_rng(200).uniform(-0.5, 0.5, 10), 8),
        ("hospital_c", np.zeros(10), np.random.default_rng(300).uniform(-0.5, 0.5, 10), 3),
    ]

    dp_updates: dict[str, tuple[np.ndarray, int]] = {}
    for hospital_id, pre, post, n in raw_updates:
        rng = np.random.default_rng(7)  # same seed for every hospital, deliberately --
        # what matters for isolating CKKS's error is that both branches below start from
        # the exact same DP-protected values, which this guarantees.
        protected = apply_dp_mechanism(pre, post, dp_config, rng)
        dp_updates[hospital_id] = (protected.dp_params, n)

    # Ground truth: plaintext weighted aggregation of the DP-protected values, no CKKS.
    plaintext_fit_results = [([values], n) for values, n in dp_updates.values()]
    plaintext_result = fedavg_aggregate(plaintext_fit_results)[0]

    # DP-protected values -> CKKS encrypt -> homomorphic aggregate -> decrypt.
    key_holder = KeyHolder.generate(ckks_config)
    specs = [ParamSpec("x", (10,), "float32", 0, 10)]
    server = EncryptedAggregationServer(key_holder.public_context_bytes, specs, round_id=1, model_version="v1")
    for hospital_id, (values, n) in dp_updates.items():
        update = encrypt_model_update(
            values, specs, key_holder.public_context_bytes, ckks_config.chunk_size, hospital_id, 1, "v1", n
        )
        server.submit_update(update)
    aggregate = server.aggregate()
    encrypted_result = key_holder.decrypt_aggregate(
        aggregate.chunk_ciphertexts, aggregate.param_specs, aggregate.total_examples
    )["x"].numpy()

    abs_errors = np.abs(plaintext_result - encrypted_result)
    max_abs_error = float(abs_errors.max())
    mean_abs_error = float(abs_errors.mean())
    relative_error = float(np.mean(abs_errors / np.maximum(np.abs(plaintext_result), 1e-12)))
    print(
        f"DP+CKKS vs plaintext-DP: max_abs_error={max_abs_error:.3e} "
        f"mean_abs_error={mean_abs_error:.3e} relative_error={relative_error:.3e}"
    )

    # This is CKKS's own precision (Module 9's numerical tolerance), not DP's noise --
    # the DP noise is identical on both sides by construction, so it cancels out.
    assert max_abs_error < 1e-3
