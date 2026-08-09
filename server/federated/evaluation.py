"""Centralized ("global") evaluation -- the primary, default evaluation strategy for
Module 7.

Runs on the server against Module 5's held-out global validation split (never a
hospital's training data, never a hospital's own local-validation carve-out). This is
what the federated round-by-round Dice/IoU is compared against the centralized baseline
with, and it is what `results.py`'s `compare_to_baseline` reads.

Distributed ("client") evaluation is a separate, optional path -- see
`hospital_nodes/client_app.py`'s `HospitalNodeClient.evaluate` -- kept clearly distinct
via `weighted_average` (`strategy.py`) never being called "global."
"""

from __future__ import annotations

from typing import Callable

from monai.data import DataLoader

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.dataset import build_dataset
from cv_model.brats.transforms import get_val_transforms
from cv_model.model import build_dice_metric, build_iou_metric, build_loss_function, build_unet_from_params
from cv_model.params import NDArrays, set_parameters
from cv_model.training.config import TrainingConfig
from cv_model.training.engine import validate
from hospital_nodes.simulation import get_global_validation_studies

EvaluateFn = Callable[[int, NDArrays, dict], "tuple[float, dict] | None"]


def weighted_average(metrics: list[tuple[int, dict]]) -> dict:
    """Sample-count-weighted mean of every numeric key present, computed per-key over
    only the clients that reported it (a hospital missing a key never dilutes it).

    Used for `fit_metrics_aggregation_fn` and, if distributed evaluation is enabled,
    `evaluate_metrics_aggregation_fn`. Never the source of "global" Dice/IoU -- that
    always comes from `build_centralized_evaluate_fn` below.
    """
    if not metrics:
        return {}
    keys = {
        key
        for _, client_metrics in metrics
        for key, value in client_metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    aggregated: dict[str, float] = {}
    for key in keys:
        contributing = [(num_examples, m[key]) for num_examples, m in metrics if key in m]
        weight_sum = sum(num_examples for num_examples, _ in contributing)
        if weight_sum == 0:
            continue
        aggregated[key] = sum(num_examples * float(value) for num_examples, value in contributing) / weight_sum
    return aggregated


def build_centralized_evaluate_fn(
    data_config: BraTSRawConfig | None = None,
    train_config: TrainingConfig | None = None,
) -> EvaluateFn:
    """Build once: discover Module 5's global validation studies and the eval DataLoader.
    Returns a closure Flower's strategy calls once per round with the aggregated global
    parameters -- never with a hospital's own training data."""
    val_studies, data_config = get_global_validation_studies(data_config)
    train_config = train_config or TrainingConfig()

    dataset = build_dataset(val_studies, get_val_transforms(data_config), data_config)
    val_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=data_config.num_workers, pin_memory=True
    )

    loss_fn = build_loss_function()
    dice_metric = build_dice_metric()
    iou_metric = build_iou_metric()

    def evaluate_fn(server_round: int, parameters: NDArrays, config: dict) -> tuple[float, dict]:
        model = build_unet_from_params(
            data_config.in_channels,
            data_config.out_channels,
            train_config.unet_channels,
            train_config.unet_strides,
            train_config.unet_num_res_units,
            train_config.device,
        )
        set_parameters(model, parameters)
        # Reuses cv_model.training.engine.validate unchanged -- the exact function
        # HospitalNode.evaluate() and the centralized baseline already use.
        metrics = validate(model, val_loader, loss_fn, train_config.device, dice_metric, iou_metric)
        return metrics.loss, {"dice": metrics.dice, "iou": metrics.iou, "num_val_studies": len(val_studies)}

    return evaluate_fn
