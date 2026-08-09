"""Federated convergence curves -- global loss/Dice/IoU vs. round. Reuses matplotlib
(already a project dependency, see `cv_model/training/plots.py`) rather than introducing a
new plotting library or an experiment-tracking platform."""

from __future__ import annotations

from pathlib import Path

from server.federated.history import FederatedHistory


def generate_federated_curves(history: FederatedHistory, plots_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    records = history.records
    if not records:
        raise ValueError("Cannot plot an empty federated history.")

    rounds = [r.round for r in records]
    saved: list[Path] = []

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rounds, [r.global_loss for r in records], marker="o")
    ax.set_xlabel("round")
    ax.set_ylabel("loss")
    ax.set_title("Federated global validation loss (centralized eval)")
    loss_path = plots_dir / "global_loss_curve.png"
    fig.savefig(loss_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(loss_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rounds, [r.global_dice for r in records], marker="o")
    ax.set_xlabel("round")
    ax.set_ylabel("global Dice (macro mean across TC/WT/ET)")
    ax.set_title("Federated global validation Dice")
    dice_path = plots_dir / "global_dice_curve.png"
    fig.savefig(dice_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(dice_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rounds, [r.global_iou for r in records], marker="o")
    ax.set_xlabel("round")
    ax.set_ylabel("global IoU (macro mean across TC/WT/ET)")
    ax.set_title("Federated global validation IoU")
    iou_path = plots_dir / "global_iou_curve.png"
    fig.savefig(iou_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(iou_path)

    return saved
