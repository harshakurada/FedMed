"""Federated experiment configuration: rounds, client participation, and local-training
overrides. `FEDMED_FED_*`-env-overridable like every other config in this project; no
value is hard-coded at its point of use.
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


def _env_optional_float(var_name: str) -> float | None:
    raw = os.environ.get(var_name)
    return None if raw is None else float(raw)


@dataclass(frozen=True)
class FederatedConfig:
    """Immutable configuration for one federated experiment."""

    num_rounds: int = field(default_factory=lambda: _env_int("FEDMED_FED_NUM_ROUNDS", 3))

    # Client participation. This project simulates exactly 3 hospitals -- these stay
    # explicit and configurable rather than hard-coded, but 1-3 is the only valid range.
    min_available_clients: int = field(default_factory=lambda: _env_int("FEDMED_FED_MIN_AVAILABLE_CLIENTS", 3))
    min_fit_clients: int = field(default_factory=lambda: _env_int("FEDMED_FED_MIN_FIT_CLIENTS", 3))
    min_evaluate_clients: int = field(default_factory=lambda: _env_int("FEDMED_FED_MIN_EVALUATE_CLIENTS", 3))
    fraction_fit: float = field(default_factory=lambda: _env_float("FEDMED_FED_FRACTION_FIT", 1.0))
    # 0.0 (default): distributed per-client evaluation is disabled -- the server's
    # centralized evaluation against Module 5's global validation set is the default and
    # recommended evaluation strategy. See server/federated/evaluation.py.
    fraction_evaluate: float = field(default_factory=lambda: _env_float("FEDMED_FED_FRACTION_EVALUATE", 0.0))

    # Local-training overrides, forwarded to every hospital's HospitalTrainingConfig.
    local_epochs: int = field(default_factory=lambda: _env_int("FEDMED_HOSPITAL_LOCAL_EPOCHS", 1))
    local_val_fraction: float = field(default_factory=lambda: _env_float("FEDMED_HOSPITAL_LOCAL_VAL_FRACTION", 0.0))

    seed: int = field(default_factory=lambda: _env_int("FEDMED_FED_SEED", 42))
    checkpoint_dir: Path = field(default_factory=lambda: _env_path("FEDMED_FED_CHECKPOINT_DIR", "./checkpoints/federated"))

    # Module 8: per-round deadline for the *live* Flower deployment (wired into
    # ServerConfig(round_timeout=...) in server/server_app.py) -- so one unreachable
    # hospital node can't block a round indefinitely. None means no deadline (Flower's
    # default). Not meaningful for the in-process orchestrator (server/federated/
    # experiment.py) -- its proxy calls are synchronous Python calls, not network
    # round-trips, so there's no wall-clock wait to bound; a dropped hospital there is
    # instead handled by catching the exception its fit() call raises (see experiment.py).
    round_timeout_seconds: float | None = field(
        default_factory=lambda: _env_optional_float("FEDMED_FED_ROUND_TIMEOUT")
    )

    # Recorded for a future live deployment (a later module, once gRPC/TLS is in scope) --
    # unused by this module's in-process round orchestrator.
    server_host: str = field(default_factory=lambda: _env_str("FEDMED_FED_SERVER_HOST", "127.0.0.1"))
    server_port: int = field(default_factory=lambda: _env_int("FEDMED_FED_SERVER_PORT", 9093))


DEFAULT_CONFIG = FederatedConfig()


class FederatedConfigError(Exception):
    """Raised when a FederatedConfig itself is invalid, before building any hospital."""


def validate_federated_config(config: FederatedConfig) -> None:
    """Check configuration values for obvious problems before touching any data or model."""
    errors: list[str] = []
    if config.num_rounds <= 0:
        errors.append(f"num_rounds must be > 0, got {config.num_rounds}")
    if config.local_epochs <= 0:
        errors.append(f"local_epochs must be > 0, got {config.local_epochs}")
    if not 0.0 < config.fraction_fit <= 1.0:
        errors.append(f"fraction_fit must be in (0, 1], got {config.fraction_fit}")
    if not 0.0 <= config.fraction_evaluate <= 1.0:
        errors.append(f"fraction_evaluate must be in [0, 1], got {config.fraction_evaluate}")
    if not 0.0 <= config.local_val_fraction < 1.0:
        errors.append(f"local_val_fraction must be in [0, 1), got {config.local_val_fraction}")
    if config.fraction_evaluate > 0.0 and config.local_val_fraction == 0.0:
        errors.append(
            "fraction_evaluate > 0 requires local_val_fraction > 0 -- otherwise every "
            "sampled hospital's evaluate() raises NotImplementedError "
            "(see hospital_nodes/client_app.py:HospitalNodeClient.evaluate)"
        )
    for name, value in (
        ("min_available_clients", config.min_available_clients),
        ("min_fit_clients", config.min_fit_clients),
        ("min_evaluate_clients", config.min_evaluate_clients),
    ):
        if not 1 <= value <= 3:
            errors.append(
                f"{name} must be between 1 and 3 (this project simulates exactly 3 hospitals), got {value}"
            )
    if errors:
        raise FederatedConfigError("Invalid federated configuration:\n  - " + "\n  - ".join(errors))
    print(f"Federated config OK: {config.num_rounds} round(s), {config.local_epochs} local epoch(s)/round")
