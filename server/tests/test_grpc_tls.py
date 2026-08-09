"""Real mutual-TLS tests against a real running FedMedCoordination gRPC server -- no
mocking of grpc/ssl internals. `running_server`/`grpc_config`/`dev_certs` (server/tests/
conftest.py) generate fresh dev certificates per test and bind an OS-assigned free port;
skipped with a clear reason if `openssl` isn't on PATH rather than silently faked.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import grpc
import pytest

from server.federated.grpc_service.certs import generate_dev_certificates
from server.federated.grpc_service.config import GrpcSecurityConfig
from server.federated.grpc_service.health_client import check_health, create_secure_channel


def test_valid_mtls_client_connects_and_receives_authenticated_identity(running_server: GrpcSecurityConfig) -> None:
    channel = create_secure_channel(running_server, "hospital_a")
    try:
        response = check_health(channel, "hospital_a", timeout=running_server.timeout_seconds)
    finally:
        channel.close()

    assert response.healthy is True
    assert response.authenticated_hospital_id == "hospital_a"


def test_response_never_contains_client_private_key_material(running_server: GrpcSecurityConfig) -> None:
    channel = create_secure_channel(running_server, "hospital_b")
    try:
        response = check_health(channel, "hospital_b", timeout=running_server.timeout_seconds)
    finally:
        channel.close()

    key_bytes = running_server.client_key_path("hospital_b").read_bytes()
    assert key_bytes not in response.SerializeToString()


def test_identity_claimed_in_request_must_match_the_certificate(running_server: GrpcSecurityConfig) -> None:
    # hospital_b's real certificate, but the request body claims to be hospital_a.
    channel = create_secure_channel(running_server, "hospital_b")
    try:
        with pytest.raises(grpc.RpcError) as exc_info:
            check_health(channel, "hospital_a", timeout=running_server.timeout_seconds)
    finally:
        channel.close()
    assert exc_info.value.code() == grpc.StatusCode.PERMISSION_DENIED


def test_client_with_a_different_ca_is_rejected(running_server: GrpcSecurityConfig, tmp_path: Path) -> None:
    other_ca_dir = tmp_path / "other_ca"
    other_certs = generate_dev_certificates(other_ca_dir, ["hospital_a"], force=True)
    unrelated_config = replace(running_server, ca_cert_path=other_certs.ca_cert, certs_dir=other_ca_dir)

    channel = create_secure_channel(unrelated_config, "hospital_a")
    try:
        with pytest.raises(grpc.RpcError):
            check_health(channel, "hospital_a", timeout=running_server.timeout_seconds)
    finally:
        channel.close()


def test_client_with_no_certificate_is_rejected_mutual_tls_enforced(running_server: GrpcSecurityConfig) -> None:
    # One-way TLS only (trusts the CA, presents no client certificate) -- must be
    # rejected since the server requires mutual TLS (require_client_auth=True).
    ca_only_credentials = grpc.ssl_channel_credentials(root_certificates=running_server.ca_cert_path.read_bytes())
    channel = grpc.secure_channel(running_server.address, ca_only_credentials)
    try:
        with pytest.raises(grpc.RpcError):
            check_health(channel, "hospital_a", timeout=running_server.timeout_seconds)
    finally:
        channel.close()


def test_plain_insecure_channel_is_rejected_by_the_tls_only_server(running_server: GrpcSecurityConfig) -> None:
    channel = grpc.insecure_channel(running_server.address)
    try:
        with pytest.raises(grpc.RpcError):
            check_health(channel, "hospital_a", timeout=running_server.timeout_seconds)
    finally:
        channel.close()


def test_timeout_against_an_unreachable_address_is_bounded(dev_certs) -> None:
    unreachable_config = GrpcSecurityConfig(
        host="127.0.0.1",
        port=1,  # nothing listens on TCP port 1
        certs_dir=dev_certs.ca_cert.parent,
        ca_cert_path=dev_certs.ca_cert,
        server_cert_path=dev_certs.server_cert,
        server_key_path=dev_certs.server_key,
        timeout_seconds=1.0,
    )
    channel = create_secure_channel(unreachable_config, "hospital_a")
    started = time.monotonic()
    try:
        with pytest.raises(grpc.RpcError) as exc_info:
            check_health(channel, "hospital_a", timeout=unreachable_config.timeout_seconds)
    finally:
        channel.close()
    elapsed = time.monotonic() - started

    assert exc_info.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
    # Configurable timeout actually bounds the wait -- not "configured and ignored".
    assert elapsed < unreachable_config.timeout_seconds + 2.0
