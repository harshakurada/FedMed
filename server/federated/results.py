"""The machine-readable federated experiment record (`results.json`) -- mirrors
`cv_model.training.results.BaselineResults`'s shape so the two are directly comparable.
Distinct from `history.py` (per-round curve data): this is the one-shot summary written
once, at the end of a run.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import flwr
import monai
import torch

from cv_model.brats.config import BraTSRawConfig
from cv_model.training.config import TrainingConfig
from cv_model.training.results import DICE_SEMANTICS, BaselineResults
from server.federated.config import FederatedConfig
from server.federated.history import FederatedHistory


@dataclass
class FederatedResults:
    experiment_name: str
    timestamp: str

    num_rounds_completed: int
    best_round: int
    best_global_dice: float
    best_global_iou: float
    best_global_loss: float
    final_global_dice: float
    final_global_iou: float

    fit_metrics_aggregation: str
    dice_semantics: str

    federated_config: dict
    model_config: dict
    training_duration_seconds: float
    device: str
    software_versions: dict

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "FederatedResults":
        return cls(**json.loads(path.read_text()))


def build_results(
    experiment_name: str,
    federated_config: FederatedConfig,
    data_config: BraTSRawConfig,
    train_config: TrainingConfig,
    history: FederatedHistory,
    training_duration_seconds: float,
) -> FederatedResults:
    records = history.records
    best = max(records, key=lambda r: r.global_dice)
    final = records[-1]
    return FederatedResults(
        experiment_name=experiment_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        num_rounds_completed=len(records),
        best_round=best.round,
        best_global_dice=best.global_dice,
        best_global_iou=best.global_iou,
        best_global_loss=best.global_loss,
        final_global_dice=final.global_dice,
        final_global_iou=final.global_iou,
        fit_metrics_aggregation="sample-count-weighted average of hospitals' fit() metrics (FedAvg weighting)",
        dice_semantics=DICE_SEMANTICS,
        federated_config={
            "num_rounds": federated_config.num_rounds,
            "min_available_clients": federated_config.min_available_clients,
            "min_fit_clients": federated_config.min_fit_clients,
            "min_evaluate_clients": federated_config.min_evaluate_clients,
            "fraction_fit": federated_config.fraction_fit,
            "fraction_evaluate": federated_config.fraction_evaluate,
            "local_epochs": federated_config.local_epochs,
            "local_val_fraction": federated_config.local_val_fraction,
        },
        model_config={
            "in_channels": data_config.in_channels,
            "out_channels": data_config.out_channels,
            "unet_channels": train_config.unet_channels,
            "unet_strides": train_config.unet_strides,
            "unet_num_res_units": train_config.unet_num_res_units,
        },
        training_duration_seconds=training_duration_seconds,
        device=str(train_config.device),
        software_versions={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "monai": monai.__version__,
            "flwr": flwr.__version__,
        },
    )


def _normalize(value: object) -> object:
    return list(value) if isinstance(value, (list, tuple)) else value


def compare_to_baseline(federated: FederatedResults, baseline_results_path: Path) -> dict:
    """Compare against Module 5's centralized baseline `results.json`. Raises if the two
    runs aren't comparable (different model architecture) -- never silently compares
    incompatible experiments."""
    if not baseline_results_path.exists():
        raise FileNotFoundError(
            f"No centralized baseline results found at {baseline_results_path}. Run "
            "`python -m cv_model.training.run_baseline` first to produce one."
        )
    baseline = BaselineResults.load(baseline_results_path)

    mismatches = [
        key
        for key in ("in_channels", "out_channels", "unet_channels", "unet_strides", "unet_num_res_units")
        if _normalize(baseline.model_config[key]) != _normalize(federated.model_config[key])
    ]
    if mismatches:
        raise ValueError(
            f"Federated and centralized-baseline model configs differ in {mismatches} -- "
            "not a valid comparison. Both must use the same architecture."
        )

    return {
        "baseline_val_dice": baseline.best_val_dice,
        "federated_global_dice": federated.best_global_dice,
        "dice_delta": federated.best_global_dice - baseline.best_val_dice,
        "baseline_val_iou": baseline.best_val_iou,
        "federated_global_iou": federated.best_global_iou,
        "iou_delta": federated.best_global_iou - baseline.best_val_iou,
    }
