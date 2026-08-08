from __future__ import annotations

import os
from dataclasses import replace

import torch

from cv_model.training.config import TrainingConfig


def test_default_config_loads_with_reasonable_defaults() -> None:
    config = TrainingConfig()
    assert config.epochs > 0
    assert config.learning_rate > 0
    assert config.optimizer_name in {"adam", "sgd"}
    assert config.scheduler_type in {"none", "step", "cosine"}


def test_device_preference_cpu_returns_cpu_device() -> None:
    config = replace(TrainingConfig(), device_preference="cpu")
    assert config.device == torch.device("cpu")


def test_device_preference_auto_never_raises() -> None:
    config = replace(TrainingConfig(), device_preference="auto")
    assert config.device.type in {"cpu", "cuda"}


def test_env_var_overrides_default(monkeypatch) -> None:
    monkeypatch.setenv("FEDMED_TRAIN_EPOCHS", "7")
    monkeypatch.setenv("FEDMED_TRAIN_LR", "0.005")
    config = TrainingConfig()
    assert config.epochs == 7
    assert config.learning_rate == 0.005


def test_grad_clip_norm_defaults_to_none_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("FEDMED_TRAIN_GRAD_CLIP_NORM", raising=False)
    config = TrainingConfig()
    assert config.grad_clip_norm is None
