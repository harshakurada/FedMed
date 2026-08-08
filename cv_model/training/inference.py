"""Small inference utility: best checkpoint -> a few validation samples -> Dice/IoU + a picture.

Deliberately does not run over the whole validation set automatically --
call `run_inference(num_samples=...)` for exactly as many as you want.
Visualization reuses the same lightweight matplotlib approach as
`cv_model.brats.inspect_slices` (Module 3) rather than introducing a new
visualization framework; it's a separate function only because it operates
on already-*transformed* patch tensors + a model prediction, not the raw
NIfTI files `inspect_slices` reads.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.dataset import build_dataset, build_split
from cv_model.brats.labels import REGION_NAMES
from cv_model.brats.transforms import get_val_transforms
from cv_model.model import build_dice_metric, build_iou_metric, build_unet_from_params
from cv_model.training.checkpoint import load_checkpoint
from cv_model.training.config import TrainingConfig


def _save_comparison_grid(study_id: str, image: torch.Tensor, label: torch.Tensor, prediction: torch.Tensor, out_path: Path) -> None:
    import matplotlib.pyplot as plt  # imported lazily, mirrors cv_model.brats.inspect_slices

    mid = image.shape[-1] // 2  # middle slice along the D axis
    flair_slice = image[0, :, :, mid].numpy()
    label_slice = label[:, :, :, mid].numpy()  # (3, H, W): TC, WT, ET
    pred_slice = prediction[:, :, :, mid].numpy()

    fig, axes = plt.subplots(1, 1 + 2 * len(REGION_NAMES), figsize=(3 * (1 + 2 * len(REGION_NAMES)), 3))
    axes[0].imshow(flair_slice.T, cmap="gray", origin="lower")
    axes[0].set_title("flair")
    axes[0].axis("off")

    for i, name in enumerate(REGION_NAMES):
        ax_gt = axes[1 + i]
        ax_gt.imshow(label_slice[i].T, cmap="viridis", origin="lower")
        ax_gt.set_title(f"{name} (GT)")
        ax_gt.axis("off")

        ax_pred = axes[1 + len(REGION_NAMES) + i]
        ax_pred.imshow(pred_slice[i].T, cmap="viridis", origin="lower")
        ax_pred.set_title(f"{name} (pred)")
        ax_pred.axis("off")

    fig.suptitle(study_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def run_inference(
    data_config: BraTSRawConfig = None,
    train_config: TrainingConfig = None,
    checkpoint_path: Path | None = None,
    num_samples: int = 2,
    out_dir: Path | None = None,
) -> None:
    data_config = replace(data_config or BraTSRawConfig(), on_incomplete_study="exclude")
    train_config = train_config or TrainingConfig()
    checkpoint_path = checkpoint_path or (train_config.checkpoint_dir / "checkpoints" / "best.pt")
    out_dir = out_dir or (train_config.checkpoint_dir / "inference")
    device = train_config.device

    state = load_checkpoint(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {state.epoch}, best_val_dice={state.best_val_dice:.4f})")

    model = build_unet_from_params(
        data_config.in_channels,
        data_config.out_channels,
        train_config.unet_channels,
        train_config.unet_strides,
        train_config.unet_num_res_units,
        device,
    )
    model.load_state_dict(state.model_state_dict)
    model.eval()

    _, split = build_split(data_config)
    sample_studies = split.val[:num_samples]
    if not sample_studies:
        raise SystemExit("No validation studies available to run inference on.")

    val_ds = build_dataset(sample_studies, get_val_transforms(data_config), data_config)
    dice_metric = build_dice_metric()
    iou_metric = build_iou_metric()

    for i, study in enumerate(sample_studies):
        item = val_ds[i]
        image, label = item["image"], item["label"]
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
        prediction = (output > 0).float().cpu().squeeze(0)

        dice_metric(y_pred=prediction.unsqueeze(0), y=label.unsqueeze(0))
        iou_metric(y_pred=prediction.unsqueeze(0), y=label.unsqueeze(0))
        dice = dice_metric.aggregate()
        iou = iou_metric.aggregate()
        dice_metric.reset()
        iou_metric.reset()

        print(f"{study.study_id}:")
        for name, d, j in zip(REGION_NAMES, dice.tolist(), iou.tolist()):
            print(f"  {name}: Dice={d:.4f} IoU={j:.4f}")

        _save_comparison_grid(study.study_id, image, label, prediction, out_dir / f"{study.study_id}.png")
