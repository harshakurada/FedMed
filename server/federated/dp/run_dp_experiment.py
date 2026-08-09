"""CLI entry point for the 3-way utility comparison (Plain FedAvg / DP FedAvg /
DP+CKKS FedAvg).

    python -m server.federated.dp.run_dp_experiment

Loads the current `DPConfig`/`CKKSConfig`/`BraTSRawConfig`/`TrainingConfig` (all
`FEDMED_*`-env-var configurable). Does NOT run automatically as a side effect of
anything else in this project. Single round, real hospitals -- see
docs/differential_privacy.md for the real measured comparison numbers and why a longer
experiment isn't run automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from server.federated.dp.comparison import run_utility_comparison
from server.federated.dp.dp_config import DEFAULT_CONFIG as DEFAULT_DP_CONFIG
from server.federated.dp.dp_config import validate_dp_config
from server.federated.dp.results import DPExperimentResults, DPRoundResult

DEFAULT_RESULTS_PATH = Path("./checkpoints/dp/results/results.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-name", default="dp_utility_comparison")
    args = parser.parse_args()

    dp_config = DEFAULT_DP_CONFIG if DEFAULT_DP_CONFIG.enabled else DEFAULT_DP_CONFIG.__class__(enabled=True)
    validate_dp_config(dp_config)

    comparison = run_utility_comparison(dp_config)

    print("=== DP UTILITY COMPARISON ===")
    print(f"privacy unit: client-level (hospital-level) -- see docs/differential_privacy.md")
    for arm in (comparison.plain_fedavg, comparison.dp_fedavg, comparison.dp_ckks_fedavg):
        epsilon = f"{arm.epsilon_max_across_hospitals:.4f}" if arm.epsilon_max_across_hospitals is not None else "n/a"
        print(
            f"{arm.name}: global_dice={arm.global_dice:.4f} global_iou={arm.global_iou:.4f} "
            f"global_loss={arm.global_loss:.4f} epsilon={epsilon} aggregation_seconds={arm.aggregation_seconds:.3f}"
        )

    results = DPExperimentResults(
        experiment_name=args.experiment_name,
        privacy_unit="client-level (hospital-level)",
        rounds=[
            DPRoundResult(
                round_id=1,
                hospital_id="all",
                clip_norm=dp_config.clip_norm,
                noise_multiplier=dp_config.noise_multiplier,
                delta=dp_config.delta,
                privacy_unit="client-level (hospital-level)",
                delta_norm_before_clip=0.0,
                delta_norm_after_clip=0.0,
                epsilon_this_round=comparison.dp_fedavg.epsilon_max_across_hospitals or 0.0,
                cumulative_epsilon=comparison.dp_fedavg.epsilon_max_across_hospitals or 0.0,
                budget_status="ok",
                train_loss=comparison.dp_fedavg.global_loss,
                train_dice=comparison.dp_fedavg.global_dice,
                train_iou=comparison.dp_fedavg.global_iou,
                local_training_seconds=0.0,
                dp_processing_seconds=0.0,
                encryption_seconds=comparison.dp_ckks_fedavg.aggregation_seconds,
            )
        ],
    )
    results.save(DEFAULT_RESULTS_PATH)
    print(f"results saved to: {DEFAULT_RESULTS_PATH}")


if __name__ == "__main__":
    main()
