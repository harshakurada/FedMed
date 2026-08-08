"""Small-subset overfitting check: a PIPELINE/DEBUGGING test, not a performance benchmark.

Trains repeatedly on a handful of studies for a handful of epochs. If the
model/loss/optimizer wiring is correct, training loss should drop
substantially -- if it doesn't, something in the pipeline (not the model's
eventual real-world accuracy) is broken. This says nothing about how the
model will perform on unseen data.

Usage:
    python -m cv_model.training.overfit_check
"""

from __future__ import annotations

from dataclasses import replace

import torch
from monai.data import DataLoader

from cv_model.brats.config import DEFAULT_CONFIG as DEFAULT_DATA_CONFIG
from cv_model.brats.dataset import build_dataset
from cv_model.brats.discovery import discover_studies
from cv_model.brats.transforms import get_val_transforms
from cv_model.model import build_dice_metric, build_iou_metric, build_loss_function, build_unet_from_params
from cv_model.training.config import DEFAULT_CONFIG as DEFAULT_TRAIN_CONFIG
from cv_model.training.engine import train_one_epoch

NUM_STUDIES = 2
NUM_EPOCHS = 15
# Smaller than the default (128, 128, 64) purely so this debug loop finishes in
# a reasonable time on a CPU-only machine -- not a claim about real training patch size.
OVERFIT_PATCH_SIZE = (64, 64, 32)
# A fixed "must drop by X% within N epochs" threshold isn't a reliable pass/fail signal --
# how much loss drops in a handful of epochs depends on hardware and random init, not just
# pipeline correctness. A consistent downward trend across epochs is the more honest signal:
# a broken pipeline (e.g. mismatched loss/label representation) produces flat or erratic loss,
# not a steadily improving one.
MIN_DECREASING_EPOCH_FRACTION = 0.7


def run_overfit_check() -> None:
    data_config = replace(
        DEFAULT_DATA_CONFIG,
        on_incomplete_study="exclude",
        patch_size=OVERFIT_PATCH_SIZE,
        batch_size=NUM_STUDIES,
    )
    train_config = replace(DEFAULT_TRAIN_CONFIG, learning_rate=1e-3)
    device = train_config.device
    print(f"Device: {device}")
    print(
        f"THIS IS A PIPELINE SANITY CHECK, NOT A PERFORMANCE BENCHMARK: "
        f"training on {NUM_STUDIES} stud(y/ies) for {NUM_EPOCHS} epochs."
    )

    studies = discover_studies(data_config).valid[:NUM_STUDIES]
    if len(studies) < NUM_STUDIES:
        raise SystemExit(f"Need {NUM_STUDIES} valid studies, found {len(studies)}.")

    # Deliberately the *validation* transforms, not training's -- `RandCropByPosNegLabeld`
    # draws a new random crop every epoch, so the model would never see the same patch
    # twice and could never demonstrate memorization. A fixed deterministic crop per
    # volume is what makes this an actual overfitting test.
    dataset = build_dataset(tuple(studies), get_val_transforms(data_config), data_config)
    loader = DataLoader(dataset, batch_size=data_config.batch_size, shuffle=True, num_workers=0)

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

    losses: list[float] = []
    for epoch in range(1, NUM_EPOCHS + 1):
        metrics = train_one_epoch(model, loader, loss_fn, optimizer, device, dice_metric, iou_metric, train_config)
        losses.append(metrics.loss)
        print(f"epoch {epoch}/{NUM_EPOCHS} loss={metrics.loss:.4f} dice={metrics.dice:.4f}")

    reduction = 1 - (losses[-1] / losses[0]) if losses[0] > 0 else 0.0
    decreasing_steps = sum(1 for prev, curr in zip(losses, losses[1:]) if curr < prev)
    decreasing_fraction = decreasing_steps / (len(losses) - 1)
    print(
        f"\nLoss: {losses[0]:.4f} -> {losses[-1]:.4f} ({reduction:.1%} reduction); "
        f"decreased in {decreasing_steps}/{len(losses) - 1} epoch-to-epoch steps"
    )
    if decreasing_fraction < MIN_DECREASING_EPOCH_FRACTION or reduction <= 0:
        print(
            f"WARNING: loss trend is not consistently downward (decreased in only "
            f"{decreasing_fraction:.0%} of steps) -- investigate before running the full baseline."
        )
    else:
        print("Pipeline can learn: loss decreased consistently on this tiny subset.")


if __name__ == "__main__":
    run_overfit_check()
