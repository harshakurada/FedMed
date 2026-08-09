"""The FedMed mTLS coordination service: client side (used by each hospital).

`create_secure_channel` always builds a mutual-TLS channel -- the client verifies the
server's certificate against the CA, and presents its own certificate for the server to
verify. There is no insecure-channel path exposed here; a caller wanting to prove
insecure communication is rejected does so directly with `grpc.insecure_channel` in a
test, not through this module.
"""

from __future__ import annotations

import logging
from pathlib import Path

import grpc

from server.federated.grpc_service import fedmed_pb2, fedmed_pb2_grpc
from server.federated.grpc_service.config import GrpcSecurityConfig

logger = logging.getLogger("fedmed.grpc.client")


def create_secure_channel(config: GrpcSecurityConfig, hospital_id: str) -> grpc.Channel:
    ca_cert = Path(config.ca_cert_path).read_bytes()
    client_key = Path(config.client_key_path(hospital_id)).read_bytes()
    client_cert = Path(config.client_cert_path(hospital_id)).read_bytes()

    credentials = grpc.ssl_channel_credentials(
        root_certificates=ca_cert,
        private_key=client_key,
        certificate_chain=client_cert,
    )
    return grpc.secure_channel(config.address, credentials)


def check_health(
    channel: grpc.Channel, hospital_id: str, timeout: float | None = None
) -> fedmed_pb2.HealthCheckResponse:
    stub = fedmed_pb2_grpc.FedMedCoordinationStub(channel)
    request = fedmed_pb2.HealthCheckRequest(hospital_id=hospital_id)
    try:
        return stub.HealthCheck(request, timeout=timeout)
    except grpc.RpcError:
        logger.warning("HealthCheck failed for %s", hospital_id)
        raise
