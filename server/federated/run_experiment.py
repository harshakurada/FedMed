"""CLI entry point for the federated experiment.

    python -m server.federated.run_experiment --experiment-name federated_v1

Loads the current `FederatedConfig`/`BraTSRawConfig`/`TrainingConfig` (all
`FEDMED_*`-env-var configurable), validates the federated config, runs
`FederatedConfig.num_rounds` rounds of real FedAvg over the 3 real hospital nodes
(in-process -- see `server/federated/experiment.py`'s module docstring for why), and
writes `history.json` + `results.json` + convergence plots + checkpoints under
`FederatedConfig.checkpoint_dir`.

Does NOT run automatically as a side effect of anything else in this project.
"""

from __future__ import annotations

import argparse

from cv_model.brats.config import DEFAULT_CONFIG as DEFAULT_DATA_CONFIG
from cv_model.training.config import DEFAULT_CONFIG as DEFAULT_TRAIN_CONFIG
from server.federated.config import DEFAULT_CONFIG as DEFAULT_FEDERATED_CONFIG
from server.federated.experiment import run_federated_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-name",
        default="federated_experiment",
        help="Name recorded in results.json (does not affect checkpoint_dir).",
    )
    args = parser.parse_args()

    results = run_federated_experiment(
        DEFAULT_FEDERATED_CONFIG,
        experiment_name=args.experiment_name,
        data_config=DEFAULT_DATA_CONFIG,
        base_train_config=DEFAULT_TRAIN_CONFIG,
    )

    print("\n=== FEDERATED EXPERIMENT RESULTS ===")
    print(
        f"rounds_completed={results.num_rounds_completed} best_round={results.best_round} "
        f"best_global_dice={results.best_global_dice:.4f} best_global_iou={results.best_global_iou:.4f}"
    )
    print(f"results saved to: {DEFAULT_FEDERATED_CONFIG.checkpoint_dir / 'metrics' / 'results.json'}")


if __name__ == "__main__":
    main()
