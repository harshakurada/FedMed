"""CLI entry point for the official centralized baseline experiment.

    python -m cv_model.training.run_baseline --experiment-name brats_centralized_v1

Loads the current `BraTSRawConfig`/`TrainingConfig` (both `FEDMED_*`-env-var
configurable, see docs/dataset.md and cv_model/training/config.py), validates
them, checks for data leakage, trains, verifies the best checkpoint, runs the
final evaluation, and writes results.json + training-curve plots -- all via
`cv_model.training.experiment.run_baseline_experiment`.

Does NOT run automatically as a side effect of anything else in this
project; it only runs when explicitly invoked this way.
"""

from __future__ import annotations

import argparse

from cv_model.brats.config import DEFAULT_CONFIG as DEFAULT_DATA_CONFIG
from cv_model.training.config import DEFAULT_CONFIG as DEFAULT_TRAIN_CONFIG
from cv_model.training.experiment import ExperimentConfig, run_baseline_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-name",
        default="brats_centralized_baseline",
        help="Name recorded in results.json (does not affect the checkpoint_dir path).",
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        name=args.experiment_name,
        data_config=DEFAULT_DATA_CONFIG,
        train_config=DEFAULT_TRAIN_CONFIG,
    )
    results = run_baseline_experiment(config)

    print("\n=== OFFICIAL CENTRALIZED BASELINE RESULTS ===")
    print(f"best_epoch={results.best_epoch} best_val_dice={results.best_val_dice:.4f} "
          f"best_val_iou={results.best_val_iou:.4f}")
    print(f"per_class_dice={results.per_class_dice}")
    print(f"results saved to: {config.train_config.checkpoint_dir / 'metrics' / 'results.json'}")


if __name__ == "__main__":
    main()
