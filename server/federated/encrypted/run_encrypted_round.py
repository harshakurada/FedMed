"""Real 3-hospital encrypted-FedAvg round: local train (unchanged Module 6/7 code) ->
encrypt -> homomorphic aggregate -> decrypt -> reconstruct, compared against plaintext
FedAvg (`flwr.server.strategy.aggregate.aggregate`, reused unchanged) on the exact same
fit results.

    python -m server.federated.encrypted.run_encrypted_round

Does not run automatically as a side effect of anything else in this project. Single
round only -- this is a smoke test proving the mechanics work, not an encrypted training
experiment. See docs/homomorphic_encryption.md's "Performance" section for measured
encryption/aggregation/decryption timing -- CKKS cost scales with parameter count, and
this project's approved technologies don't include anything that would make a full
multi-round encrypted 3D U-Net run fast.

Module 10: an optional `dp_config` clips + noises each hospital's own update (see
server/federated/dp/) before it is handed to `encrypt_model_update` below -- everything
else in this file is unchanged. `dp_config=None` (the default) is byte-for-byte the
original Module 9 behavior; this is what keeps Module 9's own tests passing unmodified.

Module 12: an optional `event_sink` (same contract as Module 11's
`server/dashboard/events.py::EventSink`, used by `server/federated/experiment.py` since
Module 11) reports this round's progress to a dashboard -- `event_sink=None` (the
default) is again byte-for-byte unchanged behavior, keeping every earlier test in this
module passing unmodified.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tenseal as ts
from flwr.server.strategy.aggregate import aggregate as fedavg_aggregate

from cv_model.brats.config import BraTSRawConfig
from cv_model.params import get_parameters
from cv_model.training.config import TrainingConfig
from hospital_nodes.simulation import create_hospital_nodes
from server.dashboard.events import DashboardEvent, EventSink, EventType
from server.federated.dp.accountant import PrivacyAccountant
from server.federated.dp.dp_config import DPConfig
from server.federated.dp.dp_update import apply_dp_mechanism
from server.federated.encrypted.aggregator import EncryptedAggregationServer
from server.federated.encrypted.ckks_config import CKKSConfig
from server.federated.encrypted.encryption import encrypt_model_update
from server.federated.encrypted.key_holder import KeyHolder
from server.federated.encrypted.results import EncryptedExperimentResults
from server.federated.encrypted.serialization import flatten_model_parameters, unflatten_model_parameters

DEFAULT_RESULTS_PATH = Path("./checkpoints/encrypted/results/results.json")


def _emit(event_sink: EventSink | None, **kwargs) -> None:
    """No-op when event_sink is None -- see server/federated/experiment.py, the same
    helper Module 11 introduced there."""
    if event_sink is not None:
        event_sink.emit(DashboardEvent(**kwargs))


def run_encrypted_round_smoke_test(
    ckks_config: CKKSConfig | None = None,
    data_config: BraTSRawConfig | None = None,
    base_train_config: TrainingConfig | None = None,
    experiment_name: str = "encrypted_round_smoke_test",
    round_id: int = 1,
    model_version: str = "v1",
    dp_config: DPConfig | None = None,
    dp_accountant: PrivacyAccountant | None = None,
    dp_rng: np.random.Generator | None = None,
    event_sink: EventSink | None = None,
) -> EncryptedExperimentResults:
    """`dp_config`/`dp_accountant`/`dp_rng` are all optional and default to the original
    Module 9 behavior (no DP). When `dp_config` is provided and enabled, pass a
    `PrivacyAccountant` too (so cumulative epsilon actually accumulates across calls --
    the accountant is never created fresh per round internally) and, for reproducible
    tests, a seeded `dp_rng`; a real experiment should leave `dp_rng=None` so an unseeded
    `np.random.default_rng()` is used (never a fixed seed in production). `event_sink`
    is optional and purely observational (Module 12) -- see module docstring.
    """
    ckks_config = ckks_config or CKKSConfig()
    train_config = base_train_config or TrainingConfig()
    dp_enabled = dp_config is not None and dp_config.enabled
    if dp_enabled and dp_accountant is None:
        raise ValueError("dp_config is enabled but no dp_accountant was provided -- pass one so epsilon accumulates.")
    rng = dp_rng if dp_rng is not None else np.random.default_rng()

    hospitals, _split = create_hospital_nodes(data_config, base_train_config=train_config, local_epochs=1)
    _emit(event_sink, event_type=EventType.ROUND_STARTED, source="server", round=round_id, payload={"num_rounds": 1})

    key_holder = KeyHolder.generate(ckks_config)
    public_context_bytes = key_holder.public_context_bytes
    _emit(
        event_sink, event_type=EventType.ENCRYPTION_UPDATED, source="server",
        payload={"ckks_enabled": True, "encryption_status": "context ready", "aggregation_mode": "encrypted (ciphertext)"},
    )

    fit_results: list[tuple[list[np.ndarray], int]] = []
    encryption_seconds = 0.0
    ciphertext_size_bytes = 0
    aggregation_server: EncryptedAggregationServer | None = None

    for hospital in hospitals:
        _emit(event_sink, event_type=EventType.CLIENT_TRAINING, source=hospital.hospital_id, round=round_id)
        pre_round_ndarrays = get_parameters(hospital.model)  # captured before fit() -- this round's starting point
        result = hospital.fit()
        post_training_ndarrays = get_parameters(hospital.model)
        flattened, specs = flatten_model_parameters(hospital.model.state_dict())
        if aggregation_server is None:
            aggregation_server = EncryptedAggregationServer(public_context_bytes, specs, round_id, model_version)

        if dp_enabled:
            pre_flat = np.concatenate([a.reshape(-1).astype(np.float64) for a in pre_round_ndarrays])
            post_flat = np.concatenate([a.reshape(-1).astype(np.float64) for a in post_training_ndarrays])
            protected = apply_dp_mechanism(pre_flat, post_flat, dp_config, rng)
            flattened = protected.dp_params
            dp_record = dp_accountant.record_round(
                hospital.hospital_id,
                round_id,
                dp_config.clip_norm,
                dp_config.noise_multiplier,
                dp_config.delta,
                protected.delta_norm_before_clip,
                protected.delta_norm_after_clip,
            )
            _emit(
                event_sink, event_type=EventType.PRIVACY_UPDATED, source=hospital.hospital_id, round=round_id,
                payload={
                    "dp_enabled": True,
                    "epsilon": dp_record.epsilon_this_round,
                    "cumulative_epsilon": dp_accountant.cumulative_epsilon(hospital.hospital_id),
                    "delta": dp_config.delta,
                    "clip_norm": dp_config.clip_norm,
                    "noise_multiplier": dp_config.noise_multiplier,
                },
            )
            # The plaintext comparison below must stay apples-to-apples with whatever
            # was actually encrypted -- otherwise max_abs_error/success would measure DP
            # noise magnitude (large, and *expected*) instead of CKKS's own precision
            # (the thing this comparison exists to isolate). Reconstruct the DP-protected
            # values back into the same NDArrays layout post_training_ndarrays already has.
            dp_state = unflatten_model_parameters(flattened, specs)
            post_training_ndarrays = [dp_state[spec.name].numpy() for spec in specs]

        start = time.perf_counter()
        update = encrypt_model_update(
            flattened,
            specs,
            public_context_bytes,
            ckks_config.chunk_size,
            hospital.hospital_id,
            round_id,
            model_version,
            result.num_examples,
        )
        encryption_seconds += time.perf_counter() - start
        ciphertext_size_bytes += sum(len(chunk) for chunk in update.chunk_ciphertexts)

        aggregation_server.submit_update(update)
        fit_results.append((post_training_ndarrays, result.num_examples))
        _emit(
            event_sink, event_type=EventType.CLIENT_TRAINING_COMPLETED, source=hospital.hospital_id, round=round_id,
            payload={
                "num_examples": result.num_examples,
                "train_loss": float(result.final_train_loss),
                "train_dice": float(result.final_train_dice),
                "train_iou": float(result.final_train_iou),
            },
        )

    assert aggregation_server is not None  # at least one hospital, guaranteed by create_hospital_nodes

    start = time.perf_counter()
    encrypted_aggregate = aggregation_server.aggregate()
    encrypted_aggregation_seconds = time.perf_counter() - start
    _emit(event_sink, event_type=EventType.GLOBAL_MODEL_UPDATED, source="server", round=round_id, payload={"aggregation_mode": "encrypted (ciphertext)"})

    start = time.perf_counter()
    reconstructed = key_holder.decrypt_aggregate(
        encrypted_aggregate.chunk_ciphertexts, encrypted_aggregate.param_specs, encrypted_aggregate.total_examples
    )
    decryption_seconds = time.perf_counter() - start

    start = time.perf_counter()
    plaintext_ndarrays = fedavg_aggregate(fit_results)
    plaintext_aggregation_seconds = time.perf_counter() - start

    plaintext_flat = np.concatenate([arr.reshape(-1).astype(np.float64) for arr in plaintext_ndarrays])
    encrypted_flat, _ = flatten_model_parameters(reconstructed)

    abs_errors = np.abs(plaintext_flat - encrypted_flat)
    max_abs_error = float(abs_errors.max())
    mean_abs_error = float(abs_errors.mean())
    denom = np.where(np.abs(plaintext_flat) > 1e-12, np.abs(plaintext_flat), 1e-12)
    relative_error = float(np.mean(abs_errors / denom))

    _emit(
        event_sink, event_type=EventType.ROUND_COMPLETED, source="server", round=round_id,
        payload={
            "round_duration_seconds": encryption_seconds + encrypted_aggregation_seconds + decryption_seconds,
            "clients_completed": len(fit_results),
            "clients_failed": len(hospitals) - len(fit_results),
            "round_status": "Completed",
        },
    )

    return EncryptedExperimentResults(
        experiment_name=experiment_name,
        tenseal_version=ts.__version__,
        poly_modulus_degree=ckks_config.poly_modulus_degree,
        coeff_mod_bit_sizes=list(ckks_config.coeff_mod_bit_sizes),
        global_scale=ckks_config.global_scale,
        chunk_size=ckks_config.chunk_size,
        num_clients=len(hospitals),
        rounds=1,
        plaintext_aggregation_seconds=plaintext_aggregation_seconds,
        encryption_seconds=encryption_seconds,
        encrypted_aggregation_seconds=encrypted_aggregation_seconds,
        decryption_seconds=decryption_seconds,
        ciphertext_size_bytes=ciphertext_size_bytes,
        public_context_size_bytes=len(public_context_bytes),
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        relative_error=relative_error,
        numerical_tolerance=ckks_config.numerical_tolerance,
        success=max_abs_error <= ckks_config.numerical_tolerance,
    )


def main() -> None:
    results = run_encrypted_round_smoke_test()

    print("=== ENCRYPTED FEDAVG SMOKE TEST ===")
    print(
        f"max_abs_error={results.max_abs_error:.6e} mean_abs_error={results.mean_abs_error:.6e} "
        f"tolerance={results.numerical_tolerance:.6e} success={results.success}"
    )
    print(
        f"encryption={results.encryption_seconds:.3f}s "
        f"encrypted_aggregation={results.encrypted_aggregation_seconds:.3f}s "
        f"decryption={results.decryption_seconds:.3f}s "
        f"(plaintext_aggregation={results.plaintext_aggregation_seconds:.6f}s)"
    )
    print(
        f"ciphertext_size_bytes={results.ciphertext_size_bytes} "
        f"public_context_size_bytes={results.public_context_size_bytes}"
    )

    results.save(DEFAULT_RESULTS_PATH)
    print(f"results saved to: {DEFAULT_RESULTS_PATH}")


if __name__ == "__main__":
    main()
