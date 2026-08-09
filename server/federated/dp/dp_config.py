"""Differential Privacy configuration -- the single documented source of DP parameters.
`FEDMED_DP_*`-env-overridable, same dataclass pattern as every other config in this
project; nothing hard-coded at its point of use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(var_name: str, default: bool) -> bool:
    raw = os.environ.get(var_name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(var_name: str, default: float) -> float:
    return float(os.environ.get(var_name, default))


def _env_optional_float(var_name: str) -> float | None:
    raw = os.environ.get(var_name)
    return None if raw is None else float(raw)


@dataclass(frozen=True)
class DPConfig:
    """Immutable DP parameters for the client-level Gaussian mechanism (see
    server/federated/dp/__init__.py for the privacy-unit statement)."""

    enabled: bool = field(default_factory=lambda: _env_bool("FEDMED_DP_ENABLED", False))

    # Clipping bound C -- the max L2 norm any single hospital's per-round update
    # (post_training_params - pre_round_global_params) is allowed to have.
    clip_norm: float = field(default_factory=lambda: _env_float("FEDMED_DP_CLIP_NORM", 1.0))

    # Gaussian noise std = noise_multiplier * clip_norm. Default 5.0 is chosen (not
    # arbitrary) so that, combined with the default delta below, the per-round epsilon
    # from accountant.py's classical Gaussian mechanism bound falls just inside that
    # bound's proven validity range (epsilon < 1) -- see accountant.py.
    noise_multiplier: float = field(default_factory=lambda: _env_float("FEDMED_DP_NOISE_MULTIPLIER", 5.0))

    # Target delta for the (epsilon, delta)-DP guarantee. Must be in (0, 1), and by
    # convention much smaller than 1/(dataset size) -- see docs/differential_privacy.md.
    delta: float = field(default_factory=lambda: _env_float("FEDMED_DP_DELTA", 1e-5))

    # None (default): no budget enforcement. Set to enable
    # PrivacyAccountant.would_exceed_budget()/record_round() raising once a hospital's
    # cumulative epsilon would exceed this value.
    max_epsilon: float | None = field(default_factory=lambda: _env_optional_float("FEDMED_DP_MAX_EPSILON"))
    budget_enforcement_enabled: bool = field(
        default_factory=lambda: _env_bool("FEDMED_DP_BUDGET_ENFORCEMENT_ENABLED", False)
    )


class DPConfigError(Exception):
    """Raised when a DPConfig itself is invalid, before touching any model or data."""


def validate_dp_config(config: DPConfig) -> None:
    """Check configuration values for obvious problems before any training/encryption."""
    errors: list[str] = []
    if not 0.0 < config.delta < 1.0:
        errors.append(f"delta must be in (0, 1), got {config.delta}")
    if config.clip_norm <= 0.0:
        errors.append(f"clip_norm must be > 0, got {config.clip_norm}")
    if config.noise_multiplier <= 0.0:
        errors.append(f"noise_multiplier must be > 0, got {config.noise_multiplier}")
    if config.max_epsilon is not None and config.max_epsilon <= 0.0:
        errors.append(f"max_epsilon must be > 0 when set, got {config.max_epsilon}")
    if errors:
        raise DPConfigError("Invalid DP configuration:\n  - " + "\n  - ".join(errors))


DEFAULT_CONFIG = DPConfig()
