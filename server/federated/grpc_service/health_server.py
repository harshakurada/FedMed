"""The FedMed mTLS coordination service: server side.

Real mutual TLS -- the server requires and verifies each client's certificate
(`grpc.ssl_server_credentials(..., require_client_auth=True)`), and reads the
authenticated hospital identity back from the verified certificate's Common Name via
`context.auth_context()` -- never trusted from the request body alone. Also exposes the
standard `grpc.health.v1.Health` service, reused from `grpcio-health-checking` (already an
installed Flower dependency) rather than reinventing generic health-check plumbing.

Explicit start/stop only (see `create_grpc_server` + `grpc.Server.start`/`.stop`) --
importing this module never starts a server.
"""

from __future__ import annotations

import logging
from concurrent import futures
from pathlib import Path

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from server.federated.grpc_service import fedmed_pb2, fedmed_pb2_grpc
from server.federated.grpc_service.config import GrpcSecurityConfig

logger = logging.getLogger("fedmed.grpc.server")

_AUTH_CONTEXT_CN_KEY = "x509_common_name"


def _authenticated_hospital_id(context: grpc.ServicerContext) -> str | None:
    """Reads the CN from the client's verified TLS certificate. None means no client
    certificate was actually presented/verified for this call."""
    values = context.auth_context().get(_AUTH_CONTEXT_CN_KEY)
    if not values:
        return None
    return values[0].decode("utf-8")


class FedMedHealthServicer(fedmed_pb2_grpc.FedMedCoordinationServicer):
    """Implements HealthCheck only. No raw MRI data, labels, or model parameters are
    ever handled by this service -- only hospital identity and health/status."""

    def HealthCheck(
        self, request: fedmed_pb2.HealthCheckRequest, context: grpc.ServicerContext
    ) -> fedmed_pb2.HealthCheckResponse:
        if not request.hospital_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "hospital_id is required")

        authenticated_id = _authenticated_hospital_id(context)
        if authenticated_id is None:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "no client certificate presented")

        if authenticated_id != request.hospital_id:
            logger.warning(
                "hospital identity mismatch: request claims %r, certificate authenticates %r",
                request.hospital_id,
                authenticated_id,
            )
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                f"hospital_id {request.hospital_id!r} does not match the authenticated certificate",
            )

        logger.info("hospital %s authenticated and healthy", authenticated_id)
        return fedmed_pb2.HealthCheckResponse(
            healthy=True,
            authenticated_hospital_id=authenticated_id,
            message=f"{authenticated_id} reachable and authenticated",
        )


def create_grpc_server(config: GrpcSecurityConfig, max_workers: int = 4) -> grpc.Server:
    """Builds (but does not start) a real mutual-TLS gRPC server for the FedMed
    coordination service, plus the standard grpc.health.v1 health service."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    fedmed_pb2_grpc.add_FedMedCoordinationServicer_to_server(FedMedHealthServicer(), server)

    health_servicer = health.HealthServicer()
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    server_key = Path(config.server_key_path).read_bytes()
    server_cert = Path(config.server_cert_path).read_bytes()
    ca_cert = Path(config.ca_cert_path).read_bytes()

    credentials = grpc.ssl_server_credentials(
        [(server_key, server_cert)],
        root_certificates=ca_cert,
        require_client_auth=True,  # real mutual TLS: reject any connection without a valid client cert
    )
    server.add_secure_port(config.address, credentials)
    logger.info("FedMed gRPC coordination server configured on %s (TLS + mutual TLS enabled)", config.address)
    return server
