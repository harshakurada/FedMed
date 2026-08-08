"""MONAI transform pipelines for the BraTS/MSD Task01_BrainTumour dataset.

Defines the label remapping from the Decathlon 4-class segmentation
(background/edema/non-enhancing/enhancing) into the 3 overlapping BraTS
evaluation regions (Tumor Core, Whole Tumor, Enhancing Tumor), plus the
train/validation preprocessing + augmentation pipelines built on top of it.
"""

from __future__ import annotations

from typing import Hashable, Mapping

import numpy as np
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandShiftIntensityd,
    Spacingd,
)

from cv_model.config import BraTSConfig


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """Remap the Decathlon Task01_BrainTumour single-channel label map into
    3 binary channels matching the official BraTS evaluation regions:

      - Channel 0: Tumor Core (TC)      = labels {2, 3}
      - Channel 1: Whole Tumor (WT)     = labels {1, 2, 3}
      - Channel 2: Enhancing Tumor (ET) = label {3}

    Decathlon label convention: 0=background, 1=edema, 2=non-enhancing
    tumor core, 3=enhancing tumor. The 3 output channels overlap (a voxel
    can be both WT and TC simultaneously), which is why training treats
    this as multi-label segmentation rather than mutually-exclusive
    multi-class segmentation.
    """

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict:
        d = dict(data)
        for key in self.keys:
            label = d[key]
            tumor_core = np.logical_or(label == 2, label == 3)
            whole_tumor = np.logical_or(tumor_core, label == 1)
            enhancing_tumor = label == 3
            d[key] = np.stack([tumor_core, whole_tumor, enhancing_tumor], axis=0).astype(np.float32)
        return d


def get_train_transforms(config: BraTSConfig) -> Compose:
    """Preprocessing + data augmentation pipeline applied during training."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=config.pixdim, mode=("bilinear", "nearest")),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            # Samples patches biased toward tumor-positive regions (pos:neg = 1:1) so the
            # model doesn't collapse to predicting all-background on the heavily
            # imbalanced BraTS volumes (tumor voxels are a small minority).
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=config.patch_size,
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_val_transforms(config: BraTSConfig) -> Compose:
    """Deterministic preprocessing pipeline applied during validation/inference (no augmentation,
    no random cropping -- validation runs on full volumes via sliding-window inference)."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=config.pixdim, mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
