"""The machine-readable baseline experiment record (`results.json`).

Distinct from `history.py` (per-epoch curve data): this is the one-shot
summary written once, at the end of a run, that Module 6 will read to
compare the centralized baseline against federated FedAvg later.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import monai
import torch

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.split import SplitResult
from cv_model.training.config import TrainingConfig
from cv_model.training.final_evaluation import FinalEvaluationResult
from cv_model.training.trainer import BaselineResult

# Multi-label overlapping regions (TC/WT/ET, see cv_model/brats/labels.py) -- not mutually
# exclusive classes -- so "background" isn't a separate reported channel the way it would be
# for single-label multi-class segmentation.
DICE_SEMANTICS = (
    "Per-region Dice (Tumor Core, Whole Tumor, Enhancing Tumor), include_background=True, "
    "reduction='mean_batch' (macro average across regions -- each region weighted equally "
    "regardless of its voxel count). Multi-label overlapping regions, not mutually-exclusive "
    "classes, so there is no separate background channel to include/exclude."
)


@dataclass
class BaselineResults:
    experiment_name: str
    timestamp: str
    kind: str  # "OFFICIAL_CENTRALIZED_BASELINE" or "DEBUG_SANITY_TEST" -- never ambiguous which

    seed: int
    dataset_root: str
    val_fraction: float
    split_seed: int
    num_train_studies: int
    num_val_studies: int

    model_config: dict
    training_config: dict
    preprocessing_config: dict

    epochs_completed: int
    best_epoch: int
    stopped_early: bool
    best_val_dice: float
    best_val_iou: float
    best_val_loss: float
    final_val_dice: float
    final_val_iou: float

    dice_semantics: str
    per_class_dice: dict[str, float]
    per_class_iou: dict[str, float]
    checkpoint_reproduced_recorded_dice: bool

    training_duration_seconds: float
    device: str
    software_versions: dict[str, str]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "BaselineResults":
        return cls(**json.loads(path.read_text()))


def build_results(
    experiment_name: str,
    kind: str,
    data_config: BraTSRawConfig,
    train_config: TrainingConfig,
    split: SplitResult,
    baseline_result: BaselineResult,
    final_eval: FinalEvaluationResult,
) -> BaselineResults:
    return BaselineResults(
        experiment_name=experiment_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        kind=kind,
        seed=train_config.seed,
        dataset_root=str(data_config.root),
        val_fraction=data_config.val_fraction,
        split_seed=split.seed,
        num_train_studies=len(split.train),
        num_val_studies=len(split.val),
        model_config={
            "in_channels": data_config.in_channels,
            "out_channels": data_config.out_channels,
            "unet_channels": train_config.unet_channels,
            "unet_strides": train_config.unet_strides,
            "unet_num_res_units": train_config.unet_num_res_units,
        },
        training_config={
            "optimizer": train_config.optimizer_name,
            "learning_rate": train_config.learning_rate,
            "weight_decay": train_config.weight_decay,
            "scheduler_type": train_config.scheduler_type,
            "batch_size": data_config.batch_size,
            "epochs_configured": train_config.epochs,
            "mixed_precision": train_config.mixed_precision,
            "num_workers": data_config.num_workers,
            "checkpoint_dir": str(train_config.checkpoint_dir),
            "early_stopping_patience": train_config.early_stopping_patience,
        },
        preprocessing_config={
            "modalities": data_config.modalities,
            "pixdim": data_config.pixdim,
            "patch_size": data_config.patch_size,
            "cache_mode": data_config.cache_mode,
        },
        epochs_completed=baseline_result.epochs_completed,
        best_epoch=baseline_result.best_epoch,
        stopped_early=baseline_result.stopped_early,
        best_val_dice=baseline_result.best_val_dice,
        best_val_iou=baseline_result.best_val_iou,
        best_val_loss=baseline_result.best_val_loss,
        final_val_dice=baseline_result.final_val_dice,
        final_val_iou=baseline_result.final_val_iou,
        dice_semantics=DICE_SEMANTICS,
        per_class_dice=final_eval.per_class_dice,
        per_class_iou=final_eval.per_class_iou,
        checkpoint_reproduced_recorded_dice=final_eval.reproduced_recorded_dice,
        training_duration_seconds=baseline_result.training_time_seconds,
        device=baseline_result.device,
        software_versions={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "monai": monai.__version__,
        },
    )
