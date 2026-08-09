"""Checkpoint saving/loading: enough state to correctly resume training.

Never saves the dataset. Checkpoints are written under `TrainingConfig.checkpoint_dir`,
which is gitignored (see `.gitignore`'s `*.pt`/`*.pth`/`checkpoints/`-style patterns) --
never commit one unless explicitly asked to later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class CheckpointState:
    epoch: int
    best_val_dice: float
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any] | None
    data_config: dict[str, Any]
    train_config: dict[str, Any]
    # None for the centralized baseline; set to a hospital's identity (e.g. "hospital_a")
    # by hospital_nodes/node.py so a checkpoint file is unambiguous even if moved out of
    # its per-hospital directory.
    hospital_id: str | None = None
    metrics: dict[str, Any] | None = None


def save_checkpoint(state: CheckpointState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": state.epoch,
            "best_val_dice": state.best_val_dice,
            "model_state_dict": state.model_state_dict,
            "optimizer_state_dict": state.optimizer_state_dict,
            "scheduler_state_dict": state.scheduler_state_dict,
            "data_config": state.data_config,
            "train_config": state.train_config,
            "hospital_id": state.hospital_id,
            "metrics": state.metrics,
        },
        path,
    )


def load_checkpoint(path: Path, map_location: torch.device | str = "cpu") -> CheckpointState:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    raw = torch.load(path, map_location=map_location, weights_only=False)
    return CheckpointState(
        epoch=raw["epoch"],
        best_val_dice=raw["best_val_dice"],
        model_state_dict=raw["model_state_dict"],
        optimizer_state_dict=raw["optimizer_state_dict"],
        scheduler_state_dict=raw.get("scheduler_state_dict"),
        data_config=raw.get("data_config", {}),
        train_config=raw.get("train_config", {}),
        hospital_id=raw.get("hospital_id"),
        metrics=raw.get("metrics"),
    )
