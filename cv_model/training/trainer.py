"""Top-level orchestration for the centralized 3D U-Net baseline.

    BraTS -> Module 3 preprocessing -> DataLoader -> 3D U-Net ->
    centralized training -> validation -> best checkpoint -> baseline metrics

No Flower/hospital-node/distributed logic here -- this is exactly the
function a future federated hospital-node client would call locally, on its
own data partition, once Module 5 exists.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.dataset import build_split, get_dataloaders
from cv_model.model import build_dice_metric, build_iou_metric, build_loss_function, build_unet_from_params
from cv_model.training.checkpoint import CheckpointState, load_checkpoint, save_checkpoint
from cv_model.training.config import TrainingConfig
from cv_model.training.engine import train_one_epoch, validate
from cv_model.training.history import EpochRecord, TrainingHistory


@dataclass
class BaselineResult:
    num_train_studies: int
    num_val_studies: int
    epochs_completed: int
    best_epoch: int
    best_val_dice: float
    best_val_iou: float
    best_val_loss: float
    final_val_dice: float
    final_val_iou: float
    training_time_seconds: float
    device: str
    stopped_early: bool


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    if config.optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9
        )
    raise ValueError(f"Unknown optimizer_name: {config.optimizer_name!r} (expected 'adam' or 'sgd')")


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
    if config.scheduler_type == "none":
        return None
    if config.scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma
        )
    if config.scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    raise ValueError(f"Unknown scheduler_type: {config.scheduler_type!r} (expected 'none', 'step', or 'cosine')")


def train_baseline(
    data_config: BraTSRawConfig = None,
    train_config: TrainingConfig = None,
) -> BaselineResult:
    data_config = data_config or BraTSRawConfig()
    train_config = train_config or TrainingConfig()
    _set_seed(train_config.seed)

    device = train_config.device
    print(f"Device: {device}")
    if train_config.mixed_precision and device.type != "cuda":
        print("Warning: mixed_precision was requested but no CUDA device is available -- running in full precision.")
    use_amp = train_config.mixed_precision and device.type == "cuda"

    _, split = build_split(data_config)
    print(split.summary())
    train_loader, val_loader = get_dataloaders(data_config)

    model = build_unet_from_params(
        data_config.in_channels,
        data_config.out_channels,
        train_config.unet_channels,
        train_config.unet_strides,
        train_config.unet_num_res_units,
        device,
    )
    loss_fn = build_loss_function()
    dice_metric = build_dice_metric()
    iou_metric = build_iou_metric()
    optimizer = build_optimizer(model, train_config)
    scheduler = build_scheduler(optimizer, train_config)
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    start_epoch = 1
    best_val_dice = -1.0
    best_val_iou = 0.0
    best_val_loss = float("inf")
    history = TrainingHistory()

    if train_config.resume_from is not None:
        state = load_checkpoint(train_config.resume_from, map_location=device)
        model.load_state_dict(state.model_state_dict)
        optimizer.load_state_dict(state.optimizer_state_dict)
        if scheduler is not None and state.scheduler_state_dict is not None:
            scheduler.load_state_dict(state.scheduler_state_dict)
        start_epoch = state.epoch + 1
        best_val_dice = state.best_val_dice
        print(f"Resumed from {train_config.resume_from} at epoch {start_epoch} (best_val_dice={best_val_dice:.4f})")

    epochs_without_improvement = 0
    stopped_early = False
    final_val_dice = best_val_dice if best_val_dice >= 0 else 0.0
    final_val_iou = 0.0
    best_epoch = start_epoch - 1
    training_start = time.time()
    last_epoch_completed = start_epoch - 1

    for epoch in range(start_epoch, train_config.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, dice_metric, iou_metric, train_config, scaler
        )

        run_validation = epoch % train_config.val_frequency == 0 or epoch == train_config.epochs
        val_metrics = (
            validate(model, val_loader, loss_fn, device, dice_metric, iou_metric) if run_validation else None
        )

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        duration = time.time() - epoch_start
        last_epoch_completed = epoch

        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=train_metrics.loss,
                val_loss=val_metrics.loss if val_metrics else None,
                train_dice=train_metrics.dice,
                val_dice=val_metrics.dice if val_metrics else None,
                train_iou=train_metrics.iou,
                val_iou=val_metrics.iou if val_metrics else None,
                learning_rate=current_lr,
                duration_seconds=duration,
            )
        )
        print(
            f"epoch {epoch}/{train_config.epochs} "
            f"train_loss={train_metrics.loss:.4f} train_dice={train_metrics.dice:.4f} "
            + (f"val_loss={val_metrics.loss:.4f} val_dice={val_metrics.dice:.4f} " if val_metrics else "")
            + f"({duration:.1f}s)"
        )

        # outputs/baseline-style layout, rooted at TrainingConfig.checkpoint_dir (Module 4's
        # existing convention) rather than a second parallel output tree: checkpoints/, history/,
        # and (added by cv_model.training.experiment) metrics/, plots/, inference/.
        checkpoints_dir = train_config.checkpoint_dir / "checkpoints"
        common_state = dict(
            epoch=epoch,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
            data_config=asdict(data_config),
            train_config=asdict(train_config),
        )
        save_checkpoint(
            CheckpointState(best_val_dice=max(best_val_dice, 0.0), **common_state), checkpoints_dir / "latest.pt"
        )

        if val_metrics is not None:
            final_val_dice = val_metrics.dice
            final_val_iou = val_metrics.iou
            if val_metrics.dice > best_val_dice:
                best_val_dice = val_metrics.dice
                best_val_iou = val_metrics.iou
                best_val_loss = val_metrics.loss
                best_epoch = epoch
                save_checkpoint(
                    CheckpointState(best_val_dice=best_val_dice, **common_state), checkpoints_dir / "best.pt"
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                train_config.early_stopping_patience is not None
                and epochs_without_improvement >= train_config.early_stopping_patience
            ):
                print(f"Early stopping: no val_dice improvement for {epochs_without_improvement} validation(s).")
                stopped_early = True
                break

    history.save(train_config.checkpoint_dir / "history" / "history.json")

    return BaselineResult(
        num_train_studies=len(split.train),
        num_val_studies=len(split.val),
        epochs_completed=last_epoch_completed,
        best_epoch=best_epoch,
        best_val_dice=max(best_val_dice, 0.0),
        best_val_iou=best_val_iou,
        best_val_loss=best_val_loss,
        final_val_dice=final_val_dice,
        final_val_iou=final_val_iou,
        training_time_seconds=time.time() - training_start,
        device=str(device),
        stopped_early=stopped_early,
    )
