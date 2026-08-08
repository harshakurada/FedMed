"""Documents (via tests, not just comments) how Dice/IoU are aggregated and how an
absent class is handled -- both explicitly called out as Module 5 requirements.
"""

from __future__ import annotations

import torch

from cv_model.brats.labels import REGION_NAMES
from cv_model.model import build_dice_metric, build_iou_metric
from cv_model.training.engine import _aggregate_metric


def test_metric_reduction_is_per_region_not_a_single_scalar() -> None:
    # reduction="mean_batch" in build_dice_metric/build_iou_metric means aggregate()
    # returns one score PER REGION (macro, unweighted) -- _aggregate_metric only
    # collapses that to a single number afterwards, for convenience in logs/history.
    dice_metric = build_dice_metric()
    pred = (torch.rand(2, len(REGION_NAMES), 6, 6, 6) > 0.5).float()
    label = (torch.rand(2, len(REGION_NAMES), 6, 6, 6) > 0.5).float()
    dice_metric(y_pred=pred, y=label)

    mean_dice, per_region_dice = _aggregate_metric(dice_metric)
    assert len(per_region_dice) == len(REGION_NAMES)
    assert abs(mean_dice - _mean(per_region_dice)) < 1e-6


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def test_region_entirely_absent_across_batch_reports_zero_not_nan() -> None:
    # If a region has zero positive voxels in BOTH prediction and ground truth for every
    # sample in the batch, MONAI's ignore_empty=True (default) has nothing left to average
    # for that region and reports 0.0 -- not NaN, and not silently dropped from the output
    # array. Documented explicitly because a caller could otherwise misread 0.0 as "the
    # model completely failed on this region" rather than "this region never appeared".
    dice_metric = build_dice_metric()
    iou_metric = build_iou_metric()
    pred = torch.zeros(2, len(REGION_NAMES), 6, 6, 6)
    label = torch.zeros(2, len(REGION_NAMES), 6, 6, 6)

    dice_metric(y_pred=pred, y=label)
    iou_metric(y_pred=pred, y=label)
    _, dice_per_region = _aggregate_metric(dice_metric)
    _, iou_per_region = _aggregate_metric(iou_metric)

    assert all(v == 0.0 for v in dice_per_region)
    assert all(v == 0.0 for v in iou_per_region)
    assert not any(v != v for v in dice_per_region)  # NaN != NaN, so this checks "no NaNs"


def test_partial_region_presence_does_not_produce_nan_or_inf() -> None:
    # One sample has the region, one doesn't -- ignore_empty=True should still produce a
    # finite score using only the sample(s) where the region is actually present.
    dice_metric = build_dice_metric()
    pred = torch.zeros(2, len(REGION_NAMES), 6, 6, 6)
    label = torch.zeros(2, len(REGION_NAMES), 6, 6, 6)
    pred[0, 0, :2, :2, :2] = 1.0
    label[0, 0, :2, :2, :2] = 1.0  # sample 0, region 0: perfect match; sample 1: absent in both

    dice_metric(y_pred=pred, y=label)
    _, per_region = _aggregate_metric(dice_metric)
    assert all(v == v and abs(v) != float("inf") for v in per_region)  # finite, no NaN/Inf
