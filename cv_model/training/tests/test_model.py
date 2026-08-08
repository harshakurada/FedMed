from __future__ import annotations

import torch

from cv_model.model import build_dice_metric, build_iou_metric, build_loss_function, build_unet_from_params
from cv_model.training.tests.conftest import IN_CHANNELS, OUT_CHANNELS, TINY_PATCH


def test_build_unet_from_params_creates_model_with_expected_io() -> None:
    model = build_unet_from_params(IN_CHANNELS, OUT_CHANNELS, channels=(4, 8), strides=(2,), num_res_units=1)
    assert isinstance(model, torch.nn.Module)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_model_output_shape_matches_input_output_contract() -> None:
    model = build_unet_from_params(IN_CHANNELS, OUT_CHANNELS, channels=(4, 8), strides=(2,), num_res_units=1)
    model.eval()
    x = torch.randn(2, IN_CHANNELS, *TINY_PATCH)
    with torch.no_grad():
        y = model(x)
    # Input [B, C, D, H, W] -> Output [B, Classes, D, H, W]: same spatial dims, C -> Classes.
    assert tuple(y.shape) == (2, OUT_CHANNELS, *TINY_PATCH)


def test_loss_is_compatible_with_multilabel_output() -> None:
    # cv_model.brats.labels produces multi-channel {0,1} float masks (TC/WT/ET overlap),
    # not class-index masks -- build_loss_function's sigmoid=True, to_onehot_y=False matches that.
    loss_fn = build_loss_function()
    outputs = torch.randn(2, OUT_CHANNELS, *TINY_PATCH, requires_grad=True)
    labels = (torch.rand(2, OUT_CHANNELS, *TINY_PATCH) > 0.5).float()
    loss = loss_fn(outputs, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_dice_and_iou_metrics_produce_per_region_scores() -> None:
    dice_metric = build_dice_metric()
    iou_metric = build_iou_metric()
    predictions = (torch.rand(2, OUT_CHANNELS, *TINY_PATCH) > 0.5).float()
    labels = (torch.rand(2, OUT_CHANNELS, *TINY_PATCH) > 0.5).float()

    dice_metric(y_pred=predictions, y=labels)
    iou_metric(y_pred=predictions, y=labels)
    dice = dice_metric.aggregate()
    iou = iou_metric.aggregate()

    assert dice.shape == (OUT_CHANNELS,)
    assert iou.shape == (OUT_CHANNELS,)
    assert torch.all((dice >= 0) & (dice <= 1))
    assert torch.all((iou >= 0) & (iou <= 1))


def test_identical_prediction_and_label_gives_perfect_dice() -> None:
    dice_metric = build_dice_metric()
    label = (torch.rand(1, OUT_CHANNELS, *TINY_PATCH) > 0.5).float()
    dice_metric(y_pred=label, y=label)
    dice = dice_metric.aggregate()
    assert torch.allclose(dice, torch.ones_like(dice), atol=1e-5)
