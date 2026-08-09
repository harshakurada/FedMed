"""One full DP + CKKS + real mutual-TLS gRPC round, with optional live dashboard events --
the composition Module 12's `server/tests/test_final_integration.py` proved works, promoted
to production code (Module 13) so it has exactly one implementation instead of two: this
module is now what that test calls, and it's also what Module 13's demo entry point
(`server/demo/run_demo.py`) calls. No new mechanism lives here -- every step below is an
already-tested function from an earlier module (`apply_dp_mechanism`, `encrypt_model_update`,
`EncryptedAggregationServer`, `create_secure_channel`/`submit_encrypted_update`,
`KeyHolder.decrypt_aggregate`); this module only orchestrates them and emits the dashboard
events Module 11's schema already defines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tenseal as ts

from cv_model.params import get_parameters
from server.dashboard.events import DashboardEvent, EventSink, EventType
from server.federated.dp.accountant import PrivacyAccountant
from server.federated.dp.dp_config import DPConfig
from server.federated.dp.dp_update import apply_dp_mechanism
from server.federated.encrypted.aggregator import EncryptedAggregationServer
from server.federated.encrypted.ckks_config import CKKSConfig
from server.federated.encrypted.encryption import encrypt_model_update
from server.federated.encrypted.key_holder import KeyHolder
from server.federated.grpc_service.config import GrpcSecurityConfig
from server.federated.grpc_service.health_client import create_secure_channel, submit_encrypted_update


@dataclass
class IntegratedRoundResult:
    submitted_hospital_ids: list[str]
    failed_hospital_ids: list[str]
    reconstructed_state_dict: dict  # name -> torch.Tensor, from KeyHolder.decrypt_aggregate


def run_integrated_round(
    hospitals,
    ckks_config: CKKSConfig,
    dp_config: DPConfig,
    accountant: PrivacyAccountant,
    rng: np.random.Generator,
    key_holder: KeyHolder,
    public_context_bytes: bytes,
    specs,
    aggregation_server: EncryptedAggregationServer,
    grpc_config: GrpcSecurityConfig,
    round_id: int = 1,
    model_version: str = "v1",
    event_sink: EventSink | None = None,
    fail_hospital_id: str | None = None,
) -> IntegratedRoundResult:
    """Local train -> DP clip+noise -> CKKS encrypt -> submit over the real mTLS gRPC
    channel -> homomorphic aggregate -> authorized decrypt. `event_sink` is optional
    (default `None` = no dashboard events, same additive convention as every other
    `event_sink` parameter in this project).

    If `fail_hospital_id` is set, that hospital's gRPC submission is made to fail (a
    simulated dropped connection) -- the round still completes with the rest, and, if an
    `event_sink` is given, the dashboard is told via `CLIENT_FAILED`.
    """

    def emit(**kwargs) -> None:
        if event_sink is not None:
            event_sink.emit(DashboardEvent(**kwargs))

    emit(event_type=EventType.ROUND_STARTED, source="server", round=round_id, payload={"num_rounds": 1})
    emit(
        event_type=EventType.ENCRYPTION_UPDATED, source="server",
        payload={"ckks_enabled": True, "tls_status": "Active", "encryption_status": "context ready"},
    )

    submitted: list[str] = []
    failed: list[str] = []

    for hospital in hospitals:
        emit(event_type=EventType.CLIENT_TRAINING, source=hospital.hospital_id, round=round_id)
        pre_round = get_parameters(hospital.model)
        result = hospital.fit()
        post_training = get_parameters(hospital.model)

        pre_flat = np.concatenate([a.reshape(-1).astype(np.float64) for a in pre_round])
        post_flat = np.concatenate([a.reshape(-1).astype(np.float64) for a in post_training])

        protected = apply_dp_mechanism(pre_flat, post_flat, dp_config, rng)

        dp_record = accountant.record_round(
            hospital.hospital_id, round_id, dp_config.clip_norm, dp_config.noise_multiplier, dp_config.delta,
            protected.delta_norm_before_clip, protected.delta_norm_after_clip,
        )
        emit(
            event_type=EventType.PRIVACY_UPDATED, source=hospital.hospital_id, round=round_id,
            payload={
                "dp_enabled": True,
                "epsilon": dp_record.epsilon_this_round,
                "cumulative_epsilon": accountant.cumulative_epsilon(hospital.hospital_id),
                "delta": dp_config.delta,
                "clip_norm": dp_config.clip_norm,
                "noise_multiplier": dp_config.noise_multiplier,
            },
        )

        update = encrypt_model_update(
            protected.dp_params, specs, public_context_bytes, ckks_config.chunk_size,
            hospital.hospital_id, round_id, model_version, result.num_examples,
        )

        try:
            if hospital.hospital_id == fail_hospital_id:
                raise ConnectionError(f"simulated dropped connection for {hospital.hospital_id}")
            channel = create_secure_channel(grpc_config, hospital.hospital_id)
            try:
                response = submit_encrypted_update(channel, update, timeout=grpc_config.timeout_seconds)
            finally:
                channel.close()
            if not response.accepted:
                raise ConnectionError(f"server rejected {hospital.hospital_id}'s update: {response.message}")
        except ConnectionError:
            failed.append(hospital.hospital_id)
            emit(
                event_type=EventType.CLIENT_FAILED, source=hospital.hospital_id, round=round_id,
                payload={"reason": "submission_failed"},
            )
            continue

        submitted.append(hospital.hospital_id)
        emit(
            event_type=EventType.CLIENT_TRAINING_COMPLETED, source=hospital.hospital_id, round=round_id,
            payload={
                "num_examples": result.num_examples,
                "train_loss": float(result.final_train_loss),
                "train_dice": float(result.final_train_dice),
                "train_iou": float(result.final_train_iou),
            },
        )

    # Structural check: the aggregation server never holds a CKKS secret key -- see
    # server/tests/test_ckks_security.py for the isolated version of this same check.
    assert not hasattr(aggregation_server, "_context")
    assert ts.context_from(aggregation_server.public_context_bytes).has_secret_key() is False

    encrypted_aggregate = aggregation_server.aggregate()
    emit(
        event_type=EventType.GLOBAL_MODEL_UPDATED, source="server", round=round_id,
        payload={"aggregation_mode": "encrypted (ciphertext)"},
    )

    reconstructed = key_holder.decrypt_aggregate(
        encrypted_aggregate.chunk_ciphertexts, encrypted_aggregate.param_specs, encrypted_aggregate.total_examples
    )

    emit(
        event_type=EventType.ROUND_COMPLETED, source="server", round=round_id,
        payload={"clients_completed": len(submitted), "clients_failed": len(failed), "round_status": "Completed"},
    )

    return IntegratedRoundResult(submitted_hospital_ids=submitted, failed_hospital_ids=failed, reconstructed_state_dict=reconstructed)
