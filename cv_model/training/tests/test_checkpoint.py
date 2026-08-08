from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cv_model.model import build_unet_from_params
from cv_model.training.checkpoint import CheckpointState, load_checkpoint, save_checkpoint
from cv_model.training.tests.conftest import IN_CHANNELS, OUT_CHANNELS


def test_checkpoint_round_trip_restores_model_state(tmp_path: Path) -> None:
    model = build_unet_from_params(IN_CHANNELS, OUT_CHANNELS, channels=(4, 8), strides=(2,), num_res_units=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    state = CheckpointState(
        epoch=5,
        best_val_dice=0.42,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=None,
        data_config={"in_channels": IN_CHANNELS},
        train_config={"epochs": 5},
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(state, path)
    assert path.exists()

    loaded = load_checkpoint(path)
    assert loaded.epoch == 5
    assert loaded.best_val_dice == pytest.approx(0.42)
    assert loaded.data_config == {"in_channels": IN_CHANNELS}

    restored_model = build_unet_from_params(IN_CHANNELS, OUT_CHANNELS, channels=(4, 8), strides=(2,), num_res_units=1)
    restored_model.load_state_dict(loaded.model_state_dict)
    for original, restored in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(original, restored)


def test_load_checkpoint_raises_clearly_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(tmp_path / "does_not_exist.pt")
