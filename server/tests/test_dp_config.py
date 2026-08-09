from __future__ import annotations

import pytest

from server.federated.dp.dp_config import DPConfig, DPConfigError, validate_dp_config


def test_default_config_is_valid() -> None:
    validate_dp_config(DPConfig(enabled=True))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"delta": 0.0},
        {"delta": -0.1},
        {"delta": 1.0},
        {"delta": 1.5},
        {"clip_norm": 0.0},
        {"clip_norm": -1.0},
        {"noise_multiplier": 0.0},
        {"noise_multiplier": -2.0},
        {"max_epsilon": 0.0},
        {"max_epsilon": -1.0},
    ],
)
def test_invalid_config_values_are_rejected_clearly(kwargs: dict) -> None:
    config = DPConfig(**kwargs)
    with pytest.raises(DPConfigError):
        validate_dp_config(config)


def test_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("FEDMED_DP_ENABLED", "true")
    monkeypatch.setenv("FEDMED_DP_CLIP_NORM", "2.5")
    monkeypatch.setenv("FEDMED_DP_NOISE_MULTIPLIER", "3.0")
    monkeypatch.setenv("FEDMED_DP_DELTA", "1e-6")
    config = DPConfig()
    assert config.enabled is True
    assert config.clip_norm == 2.5
    assert config.noise_multiplier == 3.0
    assert config.delta == 1e-6


def test_max_epsilon_defaults_to_none_no_enforcement() -> None:
    config = DPConfig()
    assert config.max_epsilon is None
    assert config.budget_enforcement_enabled is False
