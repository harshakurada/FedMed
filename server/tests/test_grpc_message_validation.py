from __future__ import annotations

import grpc
import pytest

from server.federated.grpc_service.config import GrpcSecurityConfig
from server.federated.grpc_service.health_client import check_health, create_secure_channel


def test_empty_hospital_id_is_rejected_not_silently_accepted(running_server: GrpcSecurityConfig) -> None:
    channel = create_secure_channel(running_server, "hospital_a")
    try:
        with pytest.raises(grpc.RpcError) as exc_info:
            check_health(channel, "", timeout=running_server.timeout_seconds)
    finally:
        channel.close()
    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_whitespace_only_hospital_id_does_not_match_any_certificate(running_server: GrpcSecurityConfig) -> None:
    # "   " isn't the identity hospital_a's certificate authenticates -- must be
    # rejected, not silently treated as a valid claim.
    channel = create_secure_channel(running_server, "hospital_a")
    try:
        with pytest.raises(grpc.RpcError) as exc_info:
            check_health(channel, "   ", timeout=running_server.timeout_seconds)
    finally:
        channel.close()
    assert exc_info.value.code() == grpc.StatusCode.PERMISSION_DENIED


def test_unregistered_hospital_id_is_rejected(running_server: GrpcSecurityConfig) -> None:
    # No certificate was ever issued for this identity -- create_secure_channel itself
    # fails fast (no such cert file) rather than the server silently accepting anything.
    with pytest.raises(FileNotFoundError):
        create_secure_channel(running_server, "hospital_that_does_not_exist")
