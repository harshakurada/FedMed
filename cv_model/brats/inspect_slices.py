"""Lightweight dev-time inspection utility: save a PNG of a few MRI slices + label masks.

Not a web UI (that's the React dashboard, later). Loads one study directly
via nibabel -- deliberately bypassing the MONAI transform pipeline so what
you see here is as close to the raw files on disk as possible -- and saves
one PNG grid (4 modalities + 3 label regions, one middle axial slice each)
for the caller to open in any image viewer. Never displays or logs raw
pixel *values*, only a rendered image.

Usage:
    python -m cv_model.brats.inspect_slices [--study-id BraTS20_Training_001] [--out slices.png]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np

from cv_model.brats.config import DEFAULT_CONFIG, BraTSRawConfig
from cv_model.brats.discovery import StudyRecord, discover_studies
from cv_model.brats.labels import LABEL_EDEMA, LABEL_ENHANCING, LABEL_NCR_NET, REGION_NAMES


def _middle_axial_slice(volume: np.ndarray) -> np.ndarray:
    return volume[:, :, volume.shape[2] // 2]


def _find_study(config: BraTSRawConfig, study_id: str | None) -> StudyRecord:
    # A dev-time inspection tool shouldn't fail on an unrelated incomplete study
    # elsewhere in the dataset -- only the requested study needs to be usable.
    discovery = discover_studies(replace(config, on_incomplete_study="exclude"))
    if not discovery.valid:
        raise SystemExit(f"No valid studies found under {config.root}")
    if study_id is None:
        return discovery.valid[0]
    for study in discovery.valid:
        if study.study_id == study_id:
            return study
    raise SystemExit(f"Study '{study_id}' not found among {len(discovery.valid)} valid studies.")


def save_slice_grid(study: StudyRecord, config: BraTSRawConfig, out_path: Path) -> None:
    import matplotlib.pyplot as plt  # imported lazily: only this dev utility needs it

    modality_slices = {
        modality: _middle_axial_slice(np.asanyarray(nib.load(str(path)).dataobj))
        for modality, path in study.modality_paths.items()
    }
    label_volume = np.asanyarray(nib.load(str(study.label_path)).dataobj)
    label_slice = _middle_axial_slice(label_volume)
    region_masks = {
        "Tumor Core": np.isin(label_slice, [LABEL_NCR_NET, LABEL_ENHANCING]),
        "Whole Tumor": np.isin(label_slice, [LABEL_NCR_NET, LABEL_EDEMA, LABEL_ENHANCING]),
        "Enhancing Tumor": label_slice == LABEL_ENHANCING,
    }

    num_cols = len(modality_slices) + len(region_masks)
    fig, axes = plt.subplots(1, num_cols, figsize=(3 * num_cols, 3))
    for ax, (name, sl) in zip(axes, modality_slices.items()):
        ax.imshow(sl.T, cmap="gray", origin="lower")
        ax.set_title(name)
        ax.axis("off")
    for ax, name in zip(axes[len(modality_slices):], REGION_NAMES):
        ax.imshow(region_masks[name].T, cmap="viridis", origin="lower")
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(study.study_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", default=None, help="Study ID to inspect (default: first discovered).")
    parser.add_argument("--out", default="./data/brats2020/inspection/slices.png", help="Output PNG path.")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    study = _find_study(config, args.study_id)
    save_slice_grid(study, config, Path(args.out))


if __name__ == "__main__":
    main()
