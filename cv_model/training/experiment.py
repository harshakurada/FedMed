"""The official centralized baseline experiment: pre-flight checks + orchestration.

Reuses Module 4's `train_baseline` (the actual training loop) unchanged --
this module only adds what a *recorded, comparable experiment* needs on top
of it: explicit config validation, a data-leakage check before training
starts, and (after training) checkpoint verification, final evaluation,
results/plots, and a small qualitative check -- all wired together so the
official baseline command in docs/training.md does all of it in one call.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.discovery import discover_studies
from cv_model.brats.split import split_studies
from cv_model.brats.transforms import get_val_transforms
from cv_model.training.config import TrainingConfig
from cv_model.training.final_evaluation import run_final_evaluation
from cv_model.training.history import TrainingHistory
from cv_model.training.plots import generate_training_curves
from cv_model.training.results import BaselineResults, build_results
from cv_model.training.trainer import train_baseline


class ExperimentConfigError(Exception):
    """Raised when the experiment configuration itself is invalid (before touching any data)."""


class DataLeakageError(Exception):
    """Raised when the pre-flight leakage check finds a problem. Training must not proceed."""


@dataclass(frozen=True)
class ExperimentConfig:
    """The one explicit "this is the baseline experiment" record: a name plus the two
    existing Module 3/4 configs -- not a third, competing configuration system."""

    name: str
    data_config: BraTSRawConfig
    train_config: TrainingConfig


def validate_experiment_config(config: ExperimentConfig) -> None:
    """Check configuration values for obvious problems before touching the dataset or GPU."""
    errors: list[str] = []
    if config.train_config.epochs <= 0:
        errors.append(f"epochs must be > 0, got {config.train_config.epochs}")
    if config.data_config.batch_size <= 0:
        errors.append(f"batch_size must be > 0, got {config.data_config.batch_size}")
    if not 0.0 < config.data_config.val_fraction < 1.0:
        errors.append(f"val_fraction must be in (0, 1), got {config.data_config.val_fraction}")
    if config.train_config.learning_rate <= 0:
        errors.append(f"learning_rate must be > 0, got {config.train_config.learning_rate}")
    if config.train_config.val_frequency <= 0:
        errors.append(f"val_frequency must be > 0, got {config.train_config.val_frequency}")
    if len(config.train_config.unet_strides) != len(config.train_config.unet_channels) - 1:
        errors.append(
            f"unet_strides (len={len(config.train_config.unet_strides)}) must have exactly one fewer "
            f"entry than unet_channels (len={len(config.train_config.unet_channels)})"
        )
    if config.data_config.out_channels != 3:
        errors.append(
            f"out_channels={config.data_config.out_channels} but cv_model.brats.labels always "
            "produces 3 regions (TC/WT/ET) -- this would silently mismatch the label representation"
        )
    if errors:
        raise ExperimentConfigError("Invalid experiment configuration:\n  - " + "\n  - ".join(errors))
    print(f"Experiment config OK: '{config.name}'")


def check_no_data_leakage(config: ExperimentConfig) -> None:
    """Verify, on the actual discovered dataset, that the train/val split has no overlap and
    that validation preprocessing carries no random augmentation. Raises DataLeakageError and
    does NOT proceed to training if either check fails."""
    data_config = replace(config.data_config, on_incomplete_study="exclude")
    discovery = discover_studies(data_config)
    split = split_studies(list(discovery.valid), val_fraction=data_config.val_fraction, seed=data_config.seed)

    train_ids = {s.study_id for s in split.train}
    val_ids = {s.study_id for s in split.val}
    overlap = train_ids & val_ids
    if overlap:
        raise DataLeakageError(f"{len(overlap)} study/studies appear in BOTH train and val: {sorted(overlap)}")

    val_transform = get_val_transforms(data_config)
    transform_names = [type(t).__name__ for t in val_transform.transforms]
    random_transforms = [name for name in transform_names if name.startswith("Rand")]
    if random_transforms:
        raise DataLeakageError(
            f"Validation transform pipeline contains random augmentation step(s): {random_transforms} "
            "-- validation must be deterministic."
        )

    print(
        f"Data leakage check passed: {len(train_ids)} train / {len(val_ids)} val studies, "
        f"no overlap, no random augmentation in validation transforms."
    )


def run_baseline_experiment(config: ExperimentConfig, kind: str = "OFFICIAL_CENTRALIZED_BASELINE") -> BaselineResults:
    """Run the full official baseline: validate config -> leakage check -> train -> verify
    checkpoint + final evaluation -> results.json -> training curves.

    `kind` should be "OFFICIAL_CENTRALIZED_BASELINE" for a real run, or
    "DEBUG_SANITY_TEST" if this is being called for diagnostics rather than the
    experiment Module 6 will compare federated results against -- never left ambiguous
    in the saved results.json.
    """
    validate_experiment_config(config)
    check_no_data_leakage(config)

    data_config = replace(config.data_config, on_incomplete_study="exclude")
    train_config = config.train_config

    baseline_result = train_baseline(data_config, train_config)

    discovery = discover_studies(data_config)
    split = split_studies(list(discovery.valid), val_fraction=data_config.val_fraction, seed=data_config.seed)

    best_checkpoint = train_config.checkpoint_dir / "checkpoints" / "best.pt"
    final_eval = run_final_evaluation(best_checkpoint, data_config, train_config)
    print(
        f"Checkpoint validation: loaded {best_checkpoint} fresh, re-evaluated val_dice="
        f"{final_eval.val_dice:.4f} (recorded during training: {final_eval.recorded_best_val_dice:.4f}, "
        f"reproduced={final_eval.reproduced_recorded_dice})"
    )

    results = build_results(config.name, kind, data_config, train_config, split, baseline_result, final_eval)
    results.save(train_config.checkpoint_dir / "metrics" / "results.json")

    history = TrainingHistory.load(train_config.checkpoint_dir / "history" / "history.json")
    generate_training_curves(history, train_config.checkpoint_dir / "plots")

    return results
