"""Checkpoint validation + final evaluation on the Module 3 validation split.

Two Module 5 requirements collapse into one operation here: "load the best
checkpoint into a fresh model and verify it" and "run a final evaluation
that never updates weights" are the same thing -- load fresh, run
`engine.validate()` (already `model.eval()` + `torch.no_grad()`, already
never touches the optimizer), and compare against what training itself
recorded for that checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.dataset import get_dataloaders
from cv_model.brats.labels import REGION_NAMES
from cv_model.model import build_dice_metric, build_iou_metric, build_loss_function, build_unet_from_params
from cv_model.training.checkpoint import CheckpointState, load_checkpoint
from cv_model.training.config import TrainingConfig
from cv_model.training.engine import validate

# How close the re-evaluated Dice must be to the checkpoint's recorded best_val_dice to count as
# "reproduced" -- not exactly 0 because MONAI/cuDNN kernels are not bit-identical run to run even
# with the same seed (documented, not silently ignored -- see docs/training.md).
REPRODUCIBILITY_TOLERANCE = 1e-3


class CheckpointValidationError(Exception):
    """Raised when a checkpoint fails to load, or its architecture doesn't match the current config."""


@dataclass
class FinalEvaluationResult:
    checkpoint_path: str
    checkpoint_epoch: int
    val_loss: float
    val_dice: float
    val_iou: float
    per_class_dice: dict[str, float]
    per_class_iou: dict[str, float]
    reproduced_recorded_dice: bool
    recorded_best_val_dice: float


def load_and_verify_checkpoint(
    checkpoint_path: Path,
    data_config: BraTSRawConfig,
    train_config: TrainingConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, CheckpointState]:
    """Load a checkpoint into a FRESH model instance and confirm it actually loaded correctly.

    `load_state_dict` (strict by default) already raises if any parameter is missing, extra, or
    shape-mismatched -- i.e. if the checkpoint's architecture doesn't match `train_config`'s, this
    fails loudly here rather than producing a model that silently predicts garbage.
    """
    if not checkpoint_path.exists():
        raise CheckpointValidationError(f"Checkpoint not found: {checkpoint_path}")

    state = load_checkpoint(checkpoint_path, map_location=device)
    model = build_unet_from_params(
        data_config.in_channels,
        data_config.out_channels,
        train_config.unet_channels,
        train_config.unet_strides,
        train_config.unet_num_res_units,
        device,
    )
    try:
        model.load_state_dict(state.model_state_dict)
    except RuntimeError as exc:
        raise CheckpointValidationError(
            f"Checkpoint at {checkpoint_path} does not match the current model architecture "
            f"(in_channels={data_config.in_channels}, out_channels={data_config.out_channels}, "
            f"unet_channels={train_config.unet_channels}): {exc}"
        ) from exc
    return model, state


def run_final_evaluation(
    checkpoint_path: Path,
    data_config: BraTSRawConfig = None,
    train_config: TrainingConfig = None,
) -> FinalEvaluationResult:
    """Load `checkpoint_path` fresh and evaluate it on the Module 3 validation split ONLY.

    Never trains, never touches the training split. `engine.validate()` already guarantees
    `model.eval()` + `torch.no_grad()` + no optimizer step.
    """
    data_config = data_config or BraTSRawConfig()
    train_config = train_config or TrainingConfig()
    device = train_config.device

    model, state = load_and_verify_checkpoint(checkpoint_path, data_config, train_config, device)

    _, val_loader = get_dataloaders(data_config)
    loss_fn = build_loss_function()
    metrics = validate(model, val_loader, loss_fn, device, build_dice_metric(), build_iou_metric())

    per_class_dice = dict(zip(REGION_NAMES, metrics.dice_per_region))
    per_class_iou = dict(zip(REGION_NAMES, metrics.iou_per_region))
    reproduced = abs(metrics.dice - state.best_val_dice) <= REPRODUCIBILITY_TOLERANCE

    return FinalEvaluationResult(
        checkpoint_path=str(checkpoint_path),
        checkpoint_epoch=state.epoch,
        val_loss=metrics.loss,
        val_dice=metrics.dice,
        val_iou=metrics.iou,
        per_class_dice=per_class_dice,
        per_class_iou=per_class_iou,
        reproduced_recorded_dice=reproduced,
        recorded_best_val_dice=state.best_val_dice,
    )
