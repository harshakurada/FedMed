"""Encrypted updates transported over the real mTLS gRPC channel Module 8 built --
reuses `dev_certs`/`grpc_config` from `server/tests/conftest.py` but starts its own
server here since (unlike Module 8's `running_server` fixture) this one needs an
`EncryptedAggregationServer` attached.
"""

from __future__ import annotations

import numpy as np
import pytest

from server.federated.encrypted.aggregator import EncryptedAggregationServer
from server.federated.encrypted.ckks_config import CKKSConfig
from server.federated.encrypted.encryption import encrypt_model_update
from server.federated.encrypted.key_holder import KeyHolder
from server.federated.encrypted.serialization import ParamSpec
from server.federated.grpc_service.config import GrpcSecurityConfig
from server.federated.grpc_service.health_client import create_secure_channel, submit_encrypted_update
from server.federated.grpc_service.health_server import create_grpc_server


@pytest.fixture
def ckks_config() -> CKKSConfig:
    return CKKSConfig(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60), global_scale=2.0**40, chunk_size=4)


@pytest.fixture
def encrypted_round_server(grpc_config: GrpcSecurityConfig, ckks_config: CKKSConfig):
    key_holder = KeyHolder.generate(ckks_config)
    specs = [ParamSpec("x", (3,), "float32", 0, 3)]
    aggregation_server = EncryptedAggregationServer(key_holder.public_context_bytes, specs, round_id=1, model_version="v1")

    server = create_grpc_server(grpc_config, aggregation_server=aggregation_server)
    server.start()
    try:
        yield grpc_config, key_holder, aggregation_server, specs
    finally:
        server.stop(grace=1).wait()


def _encrypt(key_holder: KeyHolder, ckks_config: CKKSConfig, specs, values, hospital_id, round_id=1, model_version="v1", num_examples=5):
    return encrypt_model_update(
        np.array(values, dtype=np.float64), specs, key_holder.public_context_bytes, ckks_config.chunk_size,
        hospital_id, round_id, model_version, num_examples,
    )


def test_encrypted_update_submitted_over_real_mtls_channel_is_accepted_and_aggregatable(
    encrypted_round_server, ckks_config: CKKSConfig
) -> None:
    grpc_config, key_holder, aggregation_server, specs = encrypted_round_server
    update = _encrypt(key_holder, ckks_config, specs, [1.0, 2.0, 3.0], "hospital_a")

    channel = create_secure_channel(grpc_config, "hospital_a")
    try:
        response = submit_encrypted_update(channel, update, timeout=grpc_config.timeout_seconds)
    finally:
        channel.close()

    assert response.accepted is True

    result = aggregation_server.aggregate()
    recovered = key_holder.decrypt_aggregate(result.chunk_ciphertexts, result.param_specs, result.total_examples)
    assert recovered["x"].numpy() == pytest.approx([1.0, 2.0, 3.0], abs=1e-3)


def test_identity_mismatch_over_grpc_is_rejected(encrypted_round_server, ckks_config: CKKSConfig) -> None:
    import grpc

    grpc_config, key_holder, _aggregation_server, specs = encrypted_round_server
    # hospital_b's real certificate, but the update claims to be hospital_a.
    update = _encrypt(key_holder, ckks_config, specs, [1.0, 2.0, 3.0], "hospital_a")

    channel = create_secure_channel(grpc_config, "hospital_b")
    try:
        with pytest.raises(grpc.RpcError) as exc_info:
            submit_encrypted_update(channel, update, timeout=grpc_config.timeout_seconds)
    finally:
        channel.close()
    assert exc_info.value.code() == grpc.StatusCode.PERMISSION_DENIED


def test_stale_round_over_grpc_is_rejected(encrypted_round_server, ckks_config: CKKSConfig) -> None:
    import grpc

    grpc_config, key_holder, _aggregation_server, specs = encrypted_round_server
    stale_update = _encrypt(key_holder, ckks_config, specs, [1.0, 2.0, 3.0], "hospital_a", round_id=999)

    channel = create_secure_channel(grpc_config, "hospital_a")
    try:
        with pytest.raises(grpc.RpcError) as exc_info:
            submit_encrypted_update(channel, stale_update, timeout=grpc_config.timeout_seconds)
    finally:
        channel.close()
    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION


def test_no_ciphertext_or_metadata_ever_reveals_raw_data(encrypted_round_server, ckks_config: CKKSConfig) -> None:
    grpc_config, key_holder, _aggregation_server, specs = encrypted_round_server
    update = _encrypt(key_holder, ckks_config, specs, [1.0, 2.0, 3.0], "hospital_a")

    channel = create_secure_channel(grpc_config, "hospital_a")
    try:
        response = submit_encrypted_update(channel, update, timeout=grpc_config.timeout_seconds)
    finally:
        channel.close()

    assert not hasattr(response, "modality_paths")
    assert isinstance(response.message, str) and len(response.message) < 200
