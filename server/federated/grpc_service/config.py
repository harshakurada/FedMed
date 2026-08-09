"""Configuration for the FedMed mTLS coordination service: host/port, certificate paths,
and timeouts. `FEDMED_GRPC_*`-env-overridable, same dataclass pattern as every other
config in this project -- no path is hard-coded at its point of use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_str(var_name: str, default: str) -> str:
    return os.environ.get(var_name, default)


def _env_int(var_name: str, default: int) -> int:
    return int(os.environ.get(var_name, default))


def _env_float(var_name: str, default: float) -> float:
    return float(os.environ.get(var_name, default))


def _env_path(var_name: str, default: str) -> Path:
    return Path(os.environ.get(var_name, default)).expanduser().resolve()


@dataclass(frozen=True)
class GrpcSecurityConfig:
    """Immutable configuration for the FedMed mTLS coordination service."""

    host: str = field(default_factory=lambda: _env_str("FEDMED_GRPC_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("FEDMED_GRPC_PORT", 50051))

    certs_dir: Path = field(default_factory=lambda: _env_path("FEDMED_GRPC_CERTS_DIR", "./certs"))

    ca_cert_path: Path = field(default_factory=lambda: _env_path("FEDMED_GRPC_CA_CERT", "./certs/ca.crt"))
    server_cert_path: Path = field(default_factory=lambda: _env_path("FEDMED_GRPC_SERVER_CERT", "./certs/server.crt"))
    server_key_path: Path = field(default_factory=lambda: _env_path("FEDMED_GRPC_SERVER_KEY", "./certs/server.key"))

    timeout_seconds: float = field(default_factory=lambda: _env_float("FEDMED_GRPC_TIMEOUT_SECONDS", 5.0))
    max_retries: int = field(default_factory=lambda: _env_int("FEDMED_GRPC_MAX_RETRIES", 3))
    retry_backoff_seconds: float = field(default_factory=lambda: _env_float("FEDMED_GRPC_RETRY_BACKOFF_SECONDS", 1.0))

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    def client_cert_path(self, hospital_id: str) -> Path:
        return self.certs_dir / f"{hospital_id}.crt"

    def client_key_path(self, hospital_id: str) -> Path:
        return self.certs_dir / f"{hospital_id}.key"


DEFAULT_CONFIG = GrpcSecurityConfig()
