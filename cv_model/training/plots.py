"""Training curves generated from saved history -- reuses matplotlib (already a Module 3
dependency, `cv_model.brats.inspect_slices`) rather than introducing a plotting library."""

from __future__ import annotations

from pathlib import Path

from cv_model.training.history import TrainingHistory


def generate_training_curves(history: TrainingHistory, plots_dir: Path) -> list[Path]:
    """Save loss / Dice / IoU curves as PNGs. Only epochs with a recorded validation pass
    (`val_frequency` may skip some) contribute points to the validation lines."""
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    records = history.records
    if not records:
        raise ValueError("Cannot plot an empty training history.")

    epochs = [r.epoch for r in records]
    val_epochs = [r.epoch for r in records if r.val_loss is not None]
    saved: list[Path] = []

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [r.train_loss for r in records], label="train loss")
    if val_epochs:
        ax.plot(val_epochs, [r.val_loss for r in records if r.val_loss is not None], label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training vs validation loss")
    ax.legend()
    loss_path = plots_dir / "loss_curve.png"
    fig.savefig(loss_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(loss_path)

    if val_epochs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(val_epochs, [r.val_dice for r in records if r.val_dice is not None])
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation Dice (macro mean across TC/WT/ET)")
        ax.set_title("Validation Dice")
        dice_path = plots_dir / "dice_curve.png"
        fig.savefig(dice_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(dice_path)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(val_epochs, [r.val_iou for r in records if r.val_iou is not None])
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation IoU (macro mean across TC/WT/ET)")
        ax.set_title("Validation IoU")
        iou_path = plots_dir / "iou_curve.png"
        fig.savefig(iou_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(iou_path)

    return saved
