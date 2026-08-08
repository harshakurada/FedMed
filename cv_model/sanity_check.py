"""Structural sanity check for the 3D U-Net, independent of the BraTS dataset.

Verifies the model builds, runs a forward pass at the configured patch
size, and that the loss/metric factories accept its output -- all
without downloading Task01_BrainTumour (~7 GB). Run this first to
confirm the architecture is wired correctly; run `train_baseline.py`
(Module 2) once you're ready to actually train on real data.

Usage:
    python -m cv_model.sanity_check
"""

from __future__ import annotations

import torch

from cv_model.config import DEFAULT_CONFIG
from cv_model.model import build_dice_metric, build_loss_function, build_unet, count_parameters


def run_sanity_check() -> None:
    config = DEFAULT_CONFIG
    print(f"Device: {config.device}")

    model = build_unet(config)
    num_params = count_parameters(model)
    print(f"Model built: {num_params:,} trainable parameters")

    # Synthetic batch shaped exactly like a real preprocessed BraTS patch:
    # (batch, modalities, D, H, W) in, (batch, regions, D, H, W) out.
    batch_size = 2
    dummy_input = torch.randn(
        batch_size, config.in_channels, *config.patch_size, device=config.device
    )
    dummy_label = (
        torch.rand(batch_size, config.out_channels, *config.patch_size, device=config.device) > 0.9
    ).float()

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (batch_size, config.out_channels, *config.patch_size)
    assert tuple(output.shape) == expected_shape, (
        f"Output shape {tuple(output.shape)} != expected {expected_shape}"
    )
    print(f"Forward pass OK: input {tuple(dummy_input.shape)} -> output {tuple(output.shape)}")

    loss_fn = build_loss_function()
    loss_value = loss_fn(output, dummy_label)
    print(f"Loss function OK: DiceLoss = {loss_value.item():.4f}")

    dice_metric = build_dice_metric()
    dice_metric(y_pred=(output > 0).float(), y=dummy_label)
    per_region_dice = dice_metric.aggregate()
    dice_metric.reset()
    region_names = ["Tumor Core", "Whole Tumor", "Enhancing Tumor"]
    for name, score in zip(region_names, per_region_dice.tolist()):
        print(f"Dice metric OK: {name} = {score:.4f}")

    print("\nAll sanity checks passed. Architecture is wired correctly.")


if __name__ == "__main__":
    run_sanity_check()
