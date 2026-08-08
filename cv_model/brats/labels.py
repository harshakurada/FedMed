"""Official BraTS segmentation label convention and the transform that applies it.

Verified against this project's actual local BraTS2020 data
(`BraTS20_Training_001_seg.nii`, inspected directly): voxel values present
are exactly {0, 1, 2, 4}.

    0 = background
    1 = NCR/NET -- necrotic and non-enhancing tumor core
    2 = ED      -- peritumoral edema
    4 = ET      -- enhancing tumor
    (3 is not used -- an artifact of the challenge's labeling history)

This is a *different* convention from `cv_model/transforms.py`'s
`ConvertToMultiChannelBasedOnBratsClassesd`, which handles the Medical
Segmentation Decathlon's Task01_BrainTumour release -- MSD relabels ET from
4 to 3 and drops the unused label, so its region groupings ({2,3} for tumor
core) would silently mis-group this dataset's raw labels if reused here.
Kept as two explicit converters rather than one "smart" one so neither
dataset's label semantics has to be guessed from the other's.

Both converters produce the same 3-channel *output* order (Tumor Core,
Whole Tumor, Enhancing Tumor), matching `cv_model.model.build_unet`'s
`out_channels=3`, so a model trained against one is shape-compatible with
manifests from the other.
"""

from __future__ import annotations

from typing import Hashable, Mapping

import numpy as np
from monai.transforms import MapTransform

LABEL_BACKGROUND = 0
LABEL_NCR_NET = 1
LABEL_EDEMA = 2
LABEL_ENHANCING = 4

ALL_LABEL_VALUES = (LABEL_BACKGROUND, LABEL_NCR_NET, LABEL_EDEMA, LABEL_ENHANCING)
REGION_NAMES = ("Tumor Core", "Whole Tumor", "Enhancing Tumor")


class ConvertBraTSLabelsd(MapTransform):
    """Remap raw BraTS {0,1,2,4} labels into the 3 official evaluation regions.

      - Channel 0: Tumor Core (TC)      = labels {1, 4}
      - Channel 1: Whole Tumor (WT)     = labels {1, 2, 4}
      - Channel 2: Enhancing Tumor (ET) = label {4}

    Channels overlap (a voxel can be both WT and TC), so this is multi-label
    segmentation, not mutually-exclusive multi-class -- matching the metric
    definitions the BraTS challenge itself scores against.
    """

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> dict:
        d = dict(data)
        for key in self.keys:
            label = d[key]
            # `EnsureChannelFirstd` leaves a singleton channel axis (1, H, W, D);
            # drop it so the 3-region stack below comes out as (3, H, W, D), not (3, 1, H, W, D).
            if label.ndim == 4 and label.shape[0] == 1:
                label = label[0]
            tumor_core = np.logical_or(label == LABEL_NCR_NET, label == LABEL_ENHANCING)
            whole_tumor = np.logical_or(tumor_core, label == LABEL_EDEMA)
            enhancing_tumor = label == LABEL_ENHANCING
            d[key] = np.stack([tumor_core, whole_tumor, enhancing_tumor], axis=0).astype(np.float32)
        return d
