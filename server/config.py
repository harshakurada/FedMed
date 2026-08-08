"""Central server networking and coordination configuration.

Values here configure the Flower SuperLink's APIs and the future WebSocket
metrics stream to the dashboard (Week 3+). No coordination or streaming
logic lives here -- just the settings that logic will read, so ports/hosts
never get hard-coded at their point of use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_str(var_name: str, default: str) -> str:
    return os.environ.get(var_name, default)


def _env_int(var_name: str, default: int) -> int:
    return int(os.environ.get(var_name, default))


@dataclass(frozen=True)
class ServerConfig:
    host: str = field(default_factory=lambda: _env_str("FEDMED_SERVER_HOST", "127.0.0.1"))

    # Flower SuperLink API ports (Fleet API for nodes, Control API for `flwr run`,
    # ServerAppIo API for the ServerApp subprocess).
    fleet_api_port: int = field(default_factory=lambda: _env_int("FEDMED_FLEET_API_PORT", 9092))
    control_api_port: int = field(default_factory=lambda: _env_int("FEDMED_CONTROL_API_PORT", 9093))
    serverappio_api_port: int = field(default_factory=lambda: _env_int("FEDMED_SERVERAPPIO_API_PORT", 9091))

    # Federated round coordination -- must match the number of hospital
    # SuperNodes actually started for a round to proceed.
    num_hospital_nodes: int = field(default_factory=lambda: _env_int("FEDMED_NUM_HOSPITAL_NODES", 3))
    num_server_rounds: int = field(default_factory=lambda: _env_int("FEDMED_NUM_SERVER_ROUNDS", 3))

    # Live metrics stream to the dashboard -- planned for Week 3, not implemented yet.
    websocket_host: str = field(default_factory=lambda: _env_str("FEDMED_WS_HOST", "127.0.0.1"))
    websocket_port: int = field(default_factory=lambda: _env_int("FEDMED_WS_PORT", 8765))

    @property
    def fleet_api_address(self) -> str:
        return f"{self.host}:{self.fleet_api_port}"

    @property
    def control_api_address(self) -> str:
        return f"{self.host}:{self.control_api_port}"


DEFAULT_CONFIG = ServerConfig()
