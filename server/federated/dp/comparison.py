"""Utility comparison: Plain FedAvg vs. DP FedAvg (no encryption) vs. DP + CKKS FedAvg --
same 3 real hospitals, same fit results, same evaluation protocol for all three arms.

Reuses, never duplicates: `hospital_nodes.simulation.create_hospital_nodes` (Module 6/7),
`flwr.server.strategy.aggregate.aggregate` (Module 7's real FedAvg weighting, for both
the plaintext and the DP-no-encryption arms), `server.federated.evaluation.
build_centralized_evaluate_fn` (Module 7's centralized-evaluation function, for real
Dice/IoU/loss per arm), and Module 9's `KeyHolder`/`encrypt_model_update`/
`EncryptedAggregationServer` for the DP+CKKS arm.

Privacy metrics (epsilon, clip norms), utility metrics (Dice/IoU/loss), and security
mechanisms (CKKS/TLS) are kept in clearly separate, clearly named fields -- never
conflated (see `UtilityComparisonResult`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from flwr.server.strategy.aggregate import aggregate as fedavg_aggregate

from cv_model.brats.config import BraTSRawConfig
from cv_model.params import get_parameters
from cv_model.training.config import TrainingConfig
from hospital_nodes.simulation import create_hospital_nodes
from server.federated.dp.accountant import PrivacyAccountant
from server.federated.dp.dp_config import DPConfig
from server.federated.dp.dp_update import apply_dp_mechanism
from server.federated.encrypted.aggregator import EncryptedAggregationServer
from server.federated.encrypted.encryption import encrypt_model_update
from server.federated.encrypted.key_holder import KeyHolder
from server.federated.encrypted.serialization import flatten_model_parameters
from server.federated.encrypted.ckks_config import CKKSConfig
from server.federated.evaluation import build_centralized_evaluate_fn


@dataclass
class ArmResult:
    """One comparison arm's result. `epsilon`/`clip_norm` are None for the arm that
    doesn't apply DP (plain FedAvg) -- never a fabricated privacy number."""

    name: str
    aggregation_seconds: float
    global_loss: float
    global_dice: float
    global_iou: float
    epsilon_max_across_hospitals: float | None
    clip_norm: float | None


@dataclass
class UtilityComparisonResult:
    plain_fedavg: ArmResult
    dp_fedavg: ArmResult
    dp_ckks_fedavg: ArmResult


def run_utility_comparison(
    dp_config: DPConfig,
    ckks_config: CKKSConfig | None = None,
    data_config: BraTSRawConfig | None = None,
    base_train_config: TrainingConfig | None = None,
    round_id: int = 1,
    seed: int = 0,
) -> UtilityComparisonResult:
    ckks_config = ckks_config or CKKSConfig()
    train_config = base_train_config or TrainingConfig()

    hospitals, _split = create_hospital_nodes(data_config, base_train_config=train_config, local_epochs=1)
    evaluate_fn = build_centralized_evaluate_fn(data_config, train_config)

    pre_round_params = {h.hospital_id: get_parameters(h.model) for h in hospitals}
    for hospital in hospitals:
        hospital.fit()

    # --- Arm 1: Plain FedAvg (Module 7's real weighting function, no DP, no encryption) ---
    plain_fit_results = [
        (get_parameters(h.model), h.num_local_train_studies) for h in hospitals
    ]
    start = time.perf_counter()
    plain_aggregate = fedavg_aggregate(plain_fit_results)
    plain_seconds = time.perf_counter() - start
    plain_loss, plain_metrics = evaluate_fn(round_id, plain_aggregate, {})
    plain_arm = ArmResult(
        name="plain_fedavg",
        aggregation_seconds=plain_seconds,
        global_loss=plain_loss,
        global_dice=plain_metrics["dice"],
        global_iou=plain_metrics["iou"],
        epsilon_max_across_hospitals=None,
        clip_norm=None,
    )

    # --- Arm 2: DP FedAvg (clip + noise, plaintext weighted average, no encryption) ---
    dp_accountant = PrivacyAccountant(dp_config.max_epsilon, dp_config.budget_enforcement_enabled)
    dp_params_by_hospital = {}
    for hospital_index, hospital in enumerate(hospitals):
        # A per-hospital seed derived from an explicit index (never Python's built-in
        # hash() on a string, which is randomized per-process and would silently break
        # the "same seed -> reproducible" guarantee across separate runs).
        rng = np.random.default_rng(seed + hospital_index)
        post_params = get_parameters(hospital.model)
        pre_flat = np.concatenate([a.reshape(-1).astype(np.float64) for a in pre_round_params[hospital.hospital_id]])
        post_flat = np.concatenate([a.reshape(-1).astype(np.float64) for a in post_params])
        protected = apply_dp_mechanism(pre_flat, post_flat, dp_config, rng)
        dp_accountant.record_round(
            hospital.hospital_id, round_id, dp_config.clip_norm, dp_config.noise_multiplier, dp_config.delta,
            protected.delta_norm_before_clip, protected.delta_norm_after_clip,
        )
        dp_params_by_hospital[hospital.hospital_id] = (protected.dp_params, post_params)

    dp_fit_results = []
    for hospital in hospitals:
        dp_flat, post_params_shape_ref = dp_params_by_hospital[hospital.hospital_id]
        reshaped = _unflatten_like(dp_flat, post_params_shape_ref)
        dp_fit_results.append((reshaped, hospital.num_local_train_studies))

    start = time.perf_counter()
    dp_aggregate = fedavg_aggregate(dp_fit_results)
    dp_seconds = time.perf_counter() - start
    dp_loss, dp_metrics = evaluate_fn(round_id, dp_aggregate, {})
    max_epsilon = max(dp_accountant.cumulative_epsilon(h.hospital_id) for h in hospitals)
    dp_arm = ArmResult(
        name="dp_fedavg",
        aggregation_seconds=dp_seconds,
        global_loss=dp_loss,
        global_dice=dp_metrics["dice"],
        global_iou=dp_metrics["iou"],
        epsilon_max_across_hospitals=max_epsilon,
        clip_norm=dp_config.clip_norm,
    )

    # --- Arm 3: DP + CKKS FedAvg (same DP-protected updates, homomorphically aggregated) ---
    key_holder = KeyHolder.generate(ckks_config)
    public_context_bytes = key_holder.public_context_bytes
    _sample_flat, sample_specs = flatten_model_parameters(hospitals[0].model.state_dict())
    aggregation_server = EncryptedAggregationServer(public_context_bytes, sample_specs, round_id, "v1")

    start = time.perf_counter()
    for hospital in hospitals:
        dp_flat, _ = dp_params_by_hospital[hospital.hospital_id]
        update = encrypt_model_update(
            dp_flat, sample_specs, public_context_bytes, ckks_config.chunk_size,
            hospital.hospital_id, round_id, "v1", hospital.num_local_train_studies,
        )
        aggregation_server.submit_update(update)
    encrypted_aggregate = aggregation_server.aggregate()
    ckks_seconds = time.perf_counter() - start

    reconstructed = key_holder.decrypt_aggregate(
        encrypted_aggregate.chunk_ciphertexts, encrypted_aggregate.param_specs, encrypted_aggregate.total_examples
    )
    reconstructed_ndarrays = [reconstructed[spec.name].numpy() for spec in sample_specs]
    ckks_loss, ckks_metrics = evaluate_fn(round_id, reconstructed_ndarrays, {})
    ckks_arm = ArmResult(
        name="dp_ckks_fedavg",
        aggregation_seconds=ckks_seconds,
        global_loss=ckks_loss,
        global_dice=ckks_metrics["dice"],
        global_iou=ckks_metrics["iou"],
        epsilon_max_across_hospitals=max_epsilon,
        clip_norm=dp_config.clip_norm,
    )

    return UtilityComparisonResult(plain_fedavg=plain_arm, dp_fedavg=dp_arm, dp_ckks_fedavg=ckks_arm)


def _unflatten_like(flat: np.ndarray, shaped_like: list[np.ndarray]) -> list[np.ndarray]:
    """Reshape a flat float64 array back into the list-of-arrays NDArrays layout, using
    `shaped_like`'s shapes/dtypes as the template."""
    result = []
    offset = 0
    for template in shaped_like:
        size = template.size
        segment = flat[offset : offset + size].reshape(template.shape).astype(template.dtype)
        result.append(segment)
        offset += size
    return result
