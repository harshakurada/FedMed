"""Synthetic (non-medical) fixtures for testing the raw BraTS pipeline's logic.

These are small random arrays shaped like BraTS volumes -- never real
patient data, and never presented as if they were. They exist purely to
exercise discovery/validation/transform/dataset code paths cheaply.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.labels import LABEL_EDEMA, LABEL_ENHANCING, LABEL_NCR_NET

SYNTHETIC_SHAPE = (16, 16, 10)


def _write_synthetic_study(root: Path, study_id: str, modalities: tuple[str, ...], seed: int) -> None:
    study_dir = root / study_id
    study_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    affine = np.eye(4)

    for modality in modalities:
        # Nonzero everywhere so CropForegroundd (foreground = nonzero) doesn't crop it all away.
        volume = rng.integers(1, 500, size=SYNTHETIC_SHAPE).astype(np.float32)
        nib.save(nib.Nifti1Image(volume, affine), str(study_dir / f"{study_id}_{modality}.nii"))

    label = np.zeros(SYNTHETIC_SHAPE, dtype=np.float32)
    label[2:6, 2:6, 2:6] = LABEL_NCR_NET
    label[6:10, 2:6, 2:6] = LABEL_EDEMA
    label[2:6, 6:10, 2:6] = LABEL_ENHANCING
    nib.save(nib.Nifti1Image(label, affine), str(study_dir / f"{study_id}_seg.nii"))


@pytest.fixture
def synthetic_dataset_root(tmp_path: Path) -> Path:
    """3 complete synthetic studies + 1 incomplete one (missing 't2'), under one root."""
    modalities = ("flair", "t1", "t1ce", "t2")
    for i in range(1, 4):
        _write_synthetic_study(tmp_path, f"Synth_{i:03d}", modalities, seed=i)

    # Incomplete on purpose: real BraTS mirrors do occasionally ship a truncated
    # study (this project's own local BraTS2020 copy has one), so discovery
    # must handle it rather than assume every study is complete.
    incomplete_dir = tmp_path / "Synth_004"
    incomplete_dir.mkdir()
    rng = np.random.default_rng(4)
    for modality in ("flair", "t1", "t1ce"):  # 't2' and no restriction on 'seg' missing too? keep seg present
        volume = rng.integers(1, 500, size=SYNTHETIC_SHAPE).astype(np.float32)
        nib.save(nib.Nifti1Image(volume, np.eye(4)), str(incomplete_dir / f"Synth_004_{modality}.nii"))
    label = np.zeros(SYNTHETIC_SHAPE, dtype=np.float32)
    nib.save(nib.Nifti1Image(label, np.eye(4)), str(incomplete_dir / "Synth_004_seg.nii"))

    return tmp_path


@pytest.fixture
def tiny_config(synthetic_dataset_root: Path) -> BraTSRawConfig:
    """A BraTSRawConfig sized to match `synthetic_dataset_root`'s tiny volumes."""
    return replace(
        BraTSRawConfig(),
        root=synthetic_dataset_root,
        pixdim=(1.0, 1.0, 1.0),
        patch_size=(8, 8, 4),
        val_fraction=1 / 3,
        seed=0,
        on_incomplete_study="exclude",
        batch_size=2,
        num_workers=0,
    )
