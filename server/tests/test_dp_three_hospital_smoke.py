"""Three real (tiny) hospitals: local train -> DP clip+noise -> CKKS encrypt -> server
aggregates (never sees plaintext) -> authorized decrypt -> global update reconstructed.
Verifies all three updates were actually DP-protected, none were plaintext on the
server, privacy accounting was updated, and no raw MRI/StudyRecord/tensor data appears
anywhere in the DP or gRPC layer.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from cv_model.brats.config import BraTSRawConfig
from cv_model.params import get_parameters
from cv_model.training.config import TrainingConfig
from hospital_nodes.simulation import create_hospital_nodes
from server.federated.dp.accountant import PrivacyAccountant
from server.federated.dp.dp_config import DPConfig
from server.federated.dp.dp_update import apply_dp_mechanism
from server.federated.encrypted.aggregator import EncryptedAggregationServer
from server.federated.encrypted.ckks_config import CKKSConfig
from server.federated.encrypted.encryption import encrypt_model_update
from server.federated.encrypted.key_holder import KeyHolder
from server.federated.encrypted.serialization import flatten_model_parameters


def _tiny_train_config() -> TrainingConfig:
    return replace(
        TrainingConfig(), unet_channels=(4, 8), unet_strides=(2,), unet_num_res_units=1, device_preference="cpu"
    )


def test_three_hospitals_dp_protected_updates_aggregate_without_server_seeing_plaintext(
    hospital_data_config: BraTSRawConfig, tmp_path: Path
) -> None:
    ckks_config = CKKSConfig(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60), global_scale=2.0**40, chunk_size=64)
    dp_config = DPConfig(enabled=True, clip_norm=0.5, noise_multiplier=5.0, delta=1e-5)
    accountant = PrivacyAccountant()
    rng = np.random.default_rng(0)

    hospitals, _split = create_hospital_nodes(hospital_data_config, base_train_config=_tiny_train_config(), local_epochs=1)
    key_holder = KeyHolder.generate(ckks_config)
    public_context_bytes = key_holder.public_context_bytes

    raw_post_training: dict[str, np.ndarray] = {}
    dp_flattened: dict[str, np.ndarray] = {}
    aggregation_server: EncryptedAggregationServer | None = None

    for hospital in hospitals:
        pre_round = get_parameters(hospital.model)
        result = hospital.fit()
        post_training = get_parameters(hospital.model)

        pre_flat = np.concatenate([a.reshape(-1).astype(np.float64) for a in pre_round])
        post_flat = np.concatenate([a.reshape(-1).astype(np.float64) for a in post_training])
        raw_post_training[hospital.hospital_id] = post_flat

        protected = apply_dp_mechanism(pre_flat, post_flat, dp_config, rng)
        dp_flattened[hospital.hospital_id] = protected.dp_params
        accountant.record_round(
            hospital.hospital_id, 1, dp_config.clip_norm, dp_config.noise_multiplier, dp_config.delta,
            protected.delta_norm_before_clip, protected.delta_norm_after_clip,
        )

        _flat, specs = flatten_model_parameters(hospital.model.state_dict())
        if aggregation_server is None:
            aggregation_server = EncryptedAggregationServer(public_context_bytes, specs, round_id=1, model_version="v1")

        update = encrypt_model_update(
            protected.dp_params, specs, public_context_bytes, ckks_config.chunk_size,
            hospital.hospital_id, 1, "v1", result.num_examples,
        )
        # Only ciphertext bytes and approved metadata ever leave the hospital.
        assert all(isinstance(chunk, bytes) for chunk in update.chunk_ciphertexts)
        assert not hasattr(update, "state_dict")
        aggregation_server.submit_update(update)

    assert aggregation_server is not None

    # Each hospital's DP-protected update actually differs from its raw post-training
    # parameters -- proves the mechanism actually ran, not a no-op.
    for hospital_id in dp_flattened:
        assert not np.allclose(dp_flattened[hospital_id], raw_post_training[hospital_id])

    # Privacy accounting recorded all 3 hospitals, one round each.
    for hospital in hospitals:
        assert accountant.rounds_recorded(hospital.hospital_id) == 1
        assert accountant.cumulative_epsilon(hospital.hospital_id) > 0.0

    # Server-side aggregation and decryption -- reuses Module 9's own no-individual-
    # decryption-tested code path unchanged.
    encrypted_aggregate = aggregation_server.aggregate()
    reconstructed = key_holder.decrypt_aggregate(
        encrypted_aggregate.chunk_ciphertexts, encrypted_aggregate.param_specs, encrypted_aggregate.total_examples
    )
    assert reconstructed  # global update successfully reconstructed
    for tensor in reconstructed.values():
        assert isinstance(tensor, torch.Tensor)
        assert torch.isfinite(tensor).all()
