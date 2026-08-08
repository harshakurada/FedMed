"""MONAI transform pipelines for the locally-supplied raw BraTS dataset.

Unlike `cv_model/transforms.py` (which loads one pre-combined 4-channel
image per case from the Decathlon release), each study here is 4 separate
single-modality NIfTI files, so the pipeline first loads and concatenates
them into one "image" tensor before the rest of the pipeline proceeds.

Expected shapes for one study, at each stage:
    after LoadImaged + EnsureChannelFirstd:  each modality (1, 240, 240, 155); label (1, 240, 240, 155)
    after ConcatItemsd:                      image (4, 240, 240, 155)
    after ConvertBraTSLabelsd:                label (3, 240, 240, 155)
    after Spacingd (pixdim=1.5mm isotropic):  image/label resampled to ~(160, 160, 103) (varies per volume)
    after RandCropByPosNegLabeld (train only): image (4, *patch_size), label (3, *patch_size), batched as `num_samples`
    after CenterSpatialCropd (val only):       image/label cropped/padded to `patch_size` for a fixed-size batch
"""

from __future__ import annotations

from monai.transforms import (
    Compose,
    ConcatItemsd,
    CropForegroundd,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandShiftIntensityd,
    ResizeWithPadOrCropd,
    Spacingd,
    SpatialPadd,
)

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.labels import ConvertBraTSLabelsd


def _load_and_combine_modalities(config: BraTSRawConfig) -> list:
    """Steps shared by train and val: load each modality + label, stack modalities into "image"."""
    modality_keys = list(config.modalities)
    return [
        LoadImaged(keys=[*modality_keys, "label"]),
        EnsureChannelFirstd(keys=[*modality_keys, "label"]),
        ConcatItemsd(keys=modality_keys, name="image", dim=0),
        DeleteItemsd(keys=modality_keys),
        ConvertBraTSLabelsd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=config.pixdim, mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
    ]


def get_train_transforms(config: BraTSRawConfig = None) -> Compose:
    """Preprocessing + augmentation applied during training. Random only where the
    augmentation is realistic for MRI (flips along anatomical axes, small intensity
    jitter) -- no rotation/elastic warps that could distort tumor geometry unrealistically."""
    config = config or BraTSRawConfig()
    return Compose(
        [
            *_load_and_combine_modalities(config),
            # Foreground-cropped volumes are sometimes smaller than patch_size in one or
            # more axes (real BraTS heads don't all resample to the same foreground extent) --
            # pad up to at least patch_size so RandCropByPosNegLabeld always has a valid ROI.
            SpatialPadd(keys=["image", "label"], spatial_size=config.patch_size),
            # Biases sampling toward tumor-positive patches (pos:neg = 1:1) so the model
            # doesn't collapse to predicting all-background on the heavily class-imbalanced volumes.
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


def get_val_transforms(config: BraTSRawConfig = None) -> Compose:
    """Deterministic preprocessing for validation -- no random augmentation, so results
    are reproducible run to run. Uses a fixed center crop/pad (not random cropping) so
    every validation volume produces one same-shape sample instead of `num_samples` random ones."""
    config = config or BraTSRawConfig()
    return Compose(
        [
            *_load_and_combine_modalities(config),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=config.patch_size),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
