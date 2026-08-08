"""Model/training-mechanics sanity check, on ONE small batch only.

Verifies the full mechanical chain -- DataLoader batch -> model forward ->
loss -> backward -> optimizer step -> metrics -- works before committing to
a real training run. No full training happens here; see `overfit_check.py`
for the (still small-scale) pipeline-learns-anything check, and `trainer.py`
for the actual baseline.

Usage:
    python -m cv_model.training.sanity_check
"""

from __future__ import annotations

from dataclasses import replace

import torch

from cv_model.brats.config import DEFAULT_CONFIG as DEFAULT_DATA_CONFIG
from cv_model.brats.dataset import get_dataloaders
from cv_model.model import build_dice_metric, build_iou_metric, build_loss_function, build_unet_from_params
from cv_model.training.config import DEFAULT_CONFIG as DEFAULT_TRAIN_CONFIG


def run_sanity_check() -> None:
    data_config = replace(DEFAULT_DATA_CONFIG, on_incomplete_study="exclude", batch_size=1)
    train_config = DEFAULT_TRAIN_CONFIG
    device = train_config.device
    print(f"Device: {device}")

    print("Building DataLoaders from the Module 3 pipeline...")
    train_loader, val_loader = get_dataloaders(data_config)
    batch = next(iter(train_loader))
    images, labels = batch["image"].to(device), batch["label"].to(device)
    print(f"  batch image shape={tuple(images.shape)} label shape={tuple(labels.shape)}")
    assert images.shape[1] == data_config.in_channels, "image channel count does not match BraTSRawConfig.in_channels"
    assert labels.shape[1] == data_config.out_channels, "label channel count does not match BraTSRawConfig.out_channels"

    print("Building model...")
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
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)

    print("Forward pass...")
    model.train()
    outputs = model(images)
    print(f"  output shape={tuple(outputs.shape)}")
    expected_shape = (images.shape[0], data_config.out_channels, *images.shape[2:])
    if tuple(outputs.shape) != expected_shape:
        raise RuntimeError(
            f"Model output shape {tuple(outputs.shape)} != expected {expected_shape}. "
            "STOPPING rather than silently reshaping -- check BraTSRawConfig.out_channels "
            "against TrainingConfig's architecture settings."
        )

    print("Loss...")
    loss = loss_fn(outputs, labels)
    print(f"  loss={loss.item():.4f}")

    print("Backward pass + optimizer step...")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("  OK")

    print("Metrics...")
    predictions = (outputs.detach() > 0).float()
    dice_metric(y_pred=predictions, y=labels)
    iou_metric(y_pred=predictions, y=labels)
    print(f"  dice(per-region)={dice_metric.aggregate().tolist()}")
    print(f"  iou(per-region)={iou_metric.aggregate().tolist()}")
    dice_metric.reset()
    iou_metric.reset()

    print("\nAll sanity checks passed. Model + training mechanics are wired correctly.")


if __name__ == "__main__":
    run_sanity_check()
