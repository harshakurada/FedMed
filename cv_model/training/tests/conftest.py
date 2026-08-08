"""Synthetic tensor fixtures for Module 4 unit tests -- no real BraTS data,
no filesystem I/O. Exercises model/training mechanics only.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cv_model.training.config import TrainingConfig

TINY_PATCH = (8, 8, 4)
IN_CHANNELS = 4
OUT_CHANNELS = 3


def _synthetic_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    image = torch.randn(batch_size, IN_CHANNELS, *TINY_PATCH, generator=generator)
    label = (torch.rand(batch_size, OUT_CHANNELS, *TINY_PATCH, generator=generator) > 0.8).float()
    return {"image": image, "label": label}


@pytest.fixture
def synthetic_loader() -> list[dict[str, torch.Tensor]]:
    """A tiny "DataLoader" (plain list of 2 batches) -- `engine.py` only needs `for batch in loader`."""
    return [_synthetic_batch(), _synthetic_batch()]


@pytest.fixture
def tiny_train_config(tmp_path) -> TrainingConfig:
    return replace(
        TrainingConfig(),
        unet_channels=(4, 8),
        unet_strides=(2,),
        unet_num_res_units=1,
        epochs=1,
        device_preference="cpu",
        checkpoint_dir=tmp_path / "checkpoints",
    )
