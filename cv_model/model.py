"""3D U-Net architecture for BraTS tumor segmentation.

Wraps MONAI's `UNet` with the exact channel/stride configuration shared
across FedMed -- federated clients must all instantiate architecturally
identical models so their weights can be aggregated by FedAvg -- and
provides matching Dice loss/metric factories for training and evaluation.
"""

from __future__ import annotations

import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.networks.nets import UNet

from cv_model.config import DEFAULT_CONFIG, BraTSConfig


def build_unet_from_params(
    in_channels: int,
    out_channels: int,
    channels: tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 2,
    device: torch.device | None = None,
) -> UNet:
    """Build the shared 3D U-Net from explicit parameters rather than a config object.

    `build_unet` below (Module 1's entry point, tied to `cv_model.config.BraTSConfig`)
    and `cv_model.training` (Module 4, which combines `cv_model.brats.config.BraTSRawConfig`
    for data shape with its own `TrainingConfig` for architecture hyperparameters) both
    call into this so the actual `UNet(...)` construction exists in exactly one place.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
    )
    return model.to(device) if device is not None else model


def build_unet(config: BraTSConfig = DEFAULT_CONFIG) -> UNet:
    """Construct the 3D U-Net used as both the centralized baseline model and the
    federated global/local model architecture.

    Every hospital node and the central server must call this with the *same*
    config so their `state_dict`s stay shape-compatible for FedAvg averaging.
    """
    return build_unet_from_params(
        config.in_channels,
        config.out_channels,
        config.unet_channels,
        config.unet_strides,
        config.unet_num_res_units,
        config.device,
    )


def build_loss_function() -> DiceLoss:
    """Dice loss over the 3 overlapping BraTS regions (TC/WT/ET).

    `sigmoid=True` + `to_onehot_y=False` because the regions overlap (a voxel
    can be both WT and TC), making this multi-label segmentation -- softmax
    (which forces mutually-exclusive classes) would be the wrong choice here.
    `squared_pred=True` and the smoothing terms match the numerically stable
    formulation MONAI recommends for 3D Dice on small tumor volumes.
    """
    return DiceLoss(
        sigmoid=True,
        to_onehot_y=False,
        squared_pred=True,
        smooth_nr=0.0,
        smooth_dr=1e-5,
    )


def build_dice_metric() -> DiceMetric:
    """Per-region Dice metric for validation, matching the loss's multi-label setup.

    `reduction="mean_batch"` returns one Dice score per region (TC/WT/ET)
    averaged across the batch, instead of collapsing all regions into a
    single scalar -- FedMed's validation/audit reporting needs per-region
    scores to compare against published BraTS baselines.
    """
    return DiceMetric(include_background=True, reduction="mean_batch")


def build_iou_metric() -> MeanIoU:
    """Per-region IoU (Jaccard) metric, matching `build_dice_metric`'s multi-label setup.

    Reported alongside Dice (not instead of it) because IoU penalizes small/partial
    overlaps more harshly -- useful as a second, stricter view of the same overlap,
    not because Dice alone is insufficient.
    """
    return MeanIoU(include_background=True, reduction="mean_batch")


def count_parameters(model: torch.nn.Module) -> int:
    """Total trainable parameter count, useful for logging/model cards."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
