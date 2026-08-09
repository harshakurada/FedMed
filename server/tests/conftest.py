"""Reuses hospital_nodes/tests/conftest.py's synthetic (non-medical) fixtures rather than
duplicating them -- Module 7's server-side tests need the same kind of small, real-shaped
BraTS-layout dataset Module 6's tests already build. Also provides Module 8's dev-cert +
running-mTLS-server fixtures used by the gRPC/TLS test suite.
"""

from __future__ import annotations

import shutil
import socket
from pathlib import Path

import pytest

from hospital_nodes.tests.conftest import (  # noqa: F401
    hospital_data_config,
    synthetic_hospital_dataset_root,
    tiny_hospital_training_config,
)
from server.federated.grpc_service.certs import CertPaths, generate_dev_certificates
from server.federated.grpc_service.config import GrpcSecurityConfig
from server.federated.grpc_service.health_server import create_grpc_server

GRPC_TEST_HOSPITAL_IDS = ["hospital_a", "hospital_b", "hospital_c"]


def _free_tcp_port() -> int:
    """Ask the OS for a currently-free localhost port -- avoids hard-coding a port
    number that might collide with something else running on the machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def dev_certs(tmp_path: Path) -> CertPaths:
    if shutil.which("openssl") is None:
        pytest.skip("openssl not found on PATH -- TLS tests require it to generate dev certificates.")
    return generate_dev_certificates(tmp_path / "certs", GRPC_TEST_HOSPITAL_IDS, force=True)


@pytest.fixture
def grpc_config(dev_certs: CertPaths, tmp_path: Path) -> GrpcSecurityConfig:
    return GrpcSecurityConfig(
        host="127.0.0.1",
        port=_free_tcp_port(),
        certs_dir=tmp_path / "certs",
        ca_cert_path=dev_certs.ca_cert,
        server_cert_path=dev_certs.server_cert,
        server_key_path=dev_certs.server_key,
        timeout_seconds=2.0,
    )


@pytest.fixture
def running_server(grpc_config: GrpcSecurityConfig):
    """A real mTLS FedMed coordination server, started and stopped around each test."""
    server = create_grpc_server(grpc_config)
    server.start()
    try:
        yield grpc_config
    finally:
        server.stop(grace=1).wait()
