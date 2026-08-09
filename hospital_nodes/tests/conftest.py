"""Synthetic (non-medical) fixtures sized for 3-way hospital partitioning tests.

Mirrors cv_model/brats/tests/conftest.py's approach (small random arrays, never real
patient data) but with enough studies (12) that partitioning across 3 hospitals after
the global train/val split still leaves each hospital a non-empty share.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.labels import LABEL_EDEMA, LABEL_ENHANCING, LABEL_NCR_NET
from cv_model.training.config import TrainingConfig
from hospital_nodes.config import build_hospital_training_config

SYNTHETIC_SHAPE = (16, 16, 10)
NUM_STUDIES = 12


def _write_synthetic_study(root: Path, study_id: str, modalities: tuple[str, ...], seed: int) -> None:
    study_dir = root / study_id
    study_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    affine = np.eye(4)

    for modality in modalities:
        volume = rng.integers(1, 500, size=SYNTHETIC_SHAPE).astype(np.float32)
        nib.save(nib.Nifti1Image(volume, affine), str(study_dir / f"{study_id}_{modality}.nii"))

    label = np.zeros(SYNTHETIC_SHAPE, dtype=np.float32)
    label[2:6, 2:6, 2:6] = LABEL_NCR_NET
    label[6:10, 2:6, 2:6] = LABEL_EDEMA
    label[2:6, 6:10, 2:6] = LABEL_ENHANCING
    nib.save(nib.Nifti1Image(label, affine), str(study_dir / f"{study_id}_seg.nii"))


@pytest.fixture
def synthetic_hospital_dataset_root(tmp_path: Path) -> Path:
    """12 complete synthetic studies -- enough for a global train/val split *and* a
    3-way hospital partition of the training share to each get a non-empty partition."""
    modalities = ("flair", "t1", "t1ce", "t2")
    for i in range(1, NUM_STUDIES + 1):
        _write_synthetic_study(tmp_path, f"Synth_{i:03d}", modalities, seed=i)
    return tmp_path


@pytest.fixture
def hospital_data_config(synthetic_hospital_dataset_root: Path) -> BraTSRawConfig:
    return replace(
        BraTSRawConfig(),
        root=synthetic_hospital_dataset_root,
        pixdim=(1.0, 1.0, 1.0),
        patch_size=(8, 8, 4),
        val_fraction=0.25,
        seed=0,
        on_incomplete_study="exclude",
        batch_size=1,
        num_workers=0,
    )


def tiny_hospital_training_config(
    partition_id: int, checkpoint_root: Path, local_epochs: int = 1, local_val_fraction: float = 0.0
):
    """A HospitalTrainingConfig with an architecture small enough for the 8x8x4 synthetic
    patch size above (the default (16,32,64,128,256)-channel U-Net has too many downsampling
    stages for such a small volume), and its checkpoint dir redirected under `checkpoint_root`
    (a pytest tmp_path) rather than the real project checkpoints/hospitals/ directory."""
    tiny_train_config = replace(
        TrainingConfig(),
        unet_channels=(4, 8),
        unet_strides=(2,),
        unet_num_res_units=1,
        device_preference="cpu",
    )
    config = build_hospital_training_config(
        partition_id,
        base_train_config=tiny_train_config,
        local_epochs=local_epochs,
        local_val_fraction=local_val_fraction,
    )
    hospital_dir = checkpoint_root / f"hospital_{chr(ord('a') + partition_id)}"
    return replace(config, train_config=replace(config.train_config, checkpoint_dir=hospital_dir))
