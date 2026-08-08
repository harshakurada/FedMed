"""Training and validation loop bodies: one epoch each, nothing else.

Kept separate from `trainer.py` (which owns the outer epoch loop,
checkpointing, and early stopping) so "what happens in one epoch" can be
unit-tested on its own with a single synthetic batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from monai.data import DataLoader
from torch.amp import GradScaler, autocast

from cv_model.training.config import TrainingConfig


@dataclass
class EpochMetrics:
    loss: float
    dice: float  # macro mean across regions -- see cv_model/training/engine.py module note below
    iou: float
    dice_per_region: tuple[float, ...] = ()
    iou_per_region: tuple[float, ...] = ()


def _aggregate_metric(metric) -> tuple[float, tuple[float, ...]]:
    """Aggregate a MONAI per-region metric (Dice/IoU) into (macro mean, per-region scores), then reset it.

    Both metrics are built with `include_background=True` (there is no separate background
    *channel* here -- TC/WT/ET are overlapping foreground regions, see cv_model.brats.labels)
    and `reduction="mean_batch"`, so `aggregate()` already returns one score per region
    (macro, unweighted); the mean here just collapses those 3 region scores to one scalar
    for logging. `ignore_empty=True` (MONAI default) means a region absent from both
    prediction and ground truth in a given sample is excluded from that sample's contribution
    -- but if a region is absent across *every* sample in the batch, MONAI has nothing left to
    average and reports 0.0 for it, not NaN and not a skipped value. A 0.0 in that case means
    "no examples of this region in this batch", not "the model failed on this region".
    """
    per_region = metric.aggregate()
    metric.reset()
    return per_region.mean().item(), tuple(per_region.tolist())


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dice_metric,
    iou_metric,
    train_config: TrainingConfig,
    scaler: GradScaler | None = None,
) -> EpochMetrics:
    """Run one training epoch. Sets `model.train()` itself; caller owns the outer loop."""
    model.train()
    use_amp = scaler is not None and scaler.is_enabled()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            if train_config.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if train_config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
            optimizer.step()

        with torch.no_grad():
            predictions = (outputs.detach() > 0).float()
            dice_metric(y_pred=predictions, y=labels)
            iou_metric(y_pred=predictions, y=labels)

        running_loss += loss.item()
        num_batches += 1
        # Drop references to this batch's tensors before the next iteration allocates more.
        del images, labels, outputs, loss

    if num_batches == 0:
        raise RuntimeError("Training DataLoader produced no batches -- check the train split is non-empty.")

    dice, dice_per_region = _aggregate_metric(dice_metric)
    iou, iou_per_region = _aggregate_metric(iou_metric)
    return EpochMetrics(
        loss=running_loss / num_batches,
        dice=dice,
        iou=iou,
        dice_per_region=dice_per_region,
        iou_per_region=iou_per_region,
    )


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    dice_metric,
    iou_metric,
) -> EpochMetrics:
    """Run one full validation pass. `model.eval()` + `torch.no_grad()` -- never updates weights,
    never uses training augmentation (the loader must have been built with `get_val_transforms`)."""
    model.eval()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        predictions = (outputs > 0).float()
        dice_metric(y_pred=predictions, y=labels)
        iou_metric(y_pred=predictions, y=labels)

        running_loss += loss.item()
        num_batches += 1
        del images, labels, outputs, loss

    if num_batches == 0:
        raise RuntimeError("Validation DataLoader produced no batches -- check the val split is non-empty.")

    dice, dice_per_region = _aggregate_metric(dice_metric)
    iou, iou_per_region = _aggregate_metric(iou_metric)
    return EpochMetrics(
        loss=running_loss / num_batches,
        dice=dice,
        iou=iou,
        dice_per_region=dice_per_region,
        iou_per_region=iou_per_region,
    )
