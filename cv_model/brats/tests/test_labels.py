from __future__ import annotations

import numpy as np

from cv_model.brats.labels import ConvertBraTSLabelsd


def test_convert_brats_labels_produces_correct_region_grouping() -> None:
    # Channel-first (1, H, W, D), as EnsureChannelFirstd would leave it.
    label = np.zeros((1, 4, 4, 1), dtype=np.float32)
    label[0, 0, 0, 0] = 1  # NCR/NET -> TC, WT
    label[0, 1, 0, 0] = 2  # ED      -> WT only
    label[0, 2, 0, 0] = 4  # ET      -> TC, WT, ET
    # (3, 0, 0, 0) stays 0 -> background, none of TC/WT/ET

    out = ConvertBraTSLabelsd(keys="label")({"label": label})["label"]

    assert out.shape == (3, 4, 4, 1)
    tumor_core, whole_tumor, enhancing = out[0], out[1], out[2]

    assert tumor_core[0, 0, 0] == 1 and tumor_core[2, 0, 0] == 1 and tumor_core[1, 0, 0] == 0
    assert whole_tumor[0, 0, 0] == 1 and whole_tumor[1, 0, 0] == 1 and whole_tumor[2, 0, 0] == 1
    assert enhancing[2, 0, 0] == 1 and enhancing[0, 0, 0] == 0 and enhancing[1, 0, 0] == 0
    assert tumor_core[3, 0, 0] == 0 and whole_tumor[3, 0, 0] == 0 and enhancing[3, 0, 0] == 0


def test_convert_brats_labels_handles_missing_channel_axis() -> None:
    # Some callers may pass (H, W, D) without the leading channel dim.
    label = np.array([[[4]]], dtype=np.float32)  # shape (1, 1, 1), value = enhancing tumor
    out = ConvertBraTSLabelsd(keys="label")({"label": label})["label"]
    assert out.shape == (3, 1, 1, 1)
    assert out[2, 0, 0, 0] == 1
