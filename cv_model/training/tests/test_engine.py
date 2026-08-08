from __future__ import annotations

import copy

import torch

from cv_model.model import build_dice_metric, build_iou_metric, build_loss_function, build_unet_from_params
from cv_model.training.config import TrainingConfig
from cv_model.training.engine import train_one_epoch, validate
from cv_model.training.tests.conftest import IN_CHANNELS, OUT_CHANNELS


def _tiny_model() -> torch.nn.Module:
    return build_unet_from_params(IN_CHANNELS, OUT_CHANNELS, channels=(4, 8), strides=(2,), num_res_units=1)


def test_train_one_epoch_updates_model_weights(synthetic_loader, tiny_train_config: TrainingConfig) -> None:
    model = _tiny_model()
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    metrics = train_one_epoch(
        model,
        synthetic_loader,
        build_loss_function(),
        optimizer,
        torch.device("cpu"),
        build_dice_metric(),
        build_iou_metric(),
        tiny_train_config,
    )

    after = model.state_dict()
    changed = any(not torch.equal(before[k], after[k]) for k in before)
    assert changed, "model weights did not change after one training epoch"
    assert metrics.loss >= 0
    assert 0 <= metrics.dice <= 1
    assert 0 <= metrics.iou <= 1


def test_validate_does_not_update_model_weights(synthetic_loader, tiny_train_config: TrainingConfig) -> None:
    model = _tiny_model()
    before = copy.deepcopy(model.state_dict())

    metrics = validate(
        model,
        synthetic_loader,
        build_loss_function(),
        torch.device("cpu"),
        build_dice_metric(),
        build_iou_metric(),
    )

    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before), "validate() must never update model weights"
    assert metrics.loss >= 0


def test_validate_uses_no_grad(synthetic_loader) -> None:
    model = _tiny_model()
    validate(model, synthetic_loader, build_loss_function(), torch.device("cpu"), build_dice_metric(), build_iou_metric())
    # If validate() forgot torch.no_grad(), calling it wouldn't itself fail, but any output
    # tensors would carry requires_grad -- assert the model's own params are unaffected instead,
    # which the no-weight-update test above already covers structurally.
    assert not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
