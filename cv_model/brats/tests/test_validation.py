from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.discovery import discover_studies
from cv_model.brats.validation import validate_studies


def test_validate_studies_reports_no_issues_for_clean_data(tiny_config: BraTSRawConfig) -> None:
    result = discover_studies(tiny_config)
    report = validate_studies(list(result.valid), tiny_config, sample_size=len(result.valid))
    assert report.ok, report.summary()
    assert report.studies_checked == 3


def test_validate_studies_flags_unexpected_label_value(tiny_config: BraTSRawConfig, tmp_path: Path) -> None:
    result = discover_studies(tiny_config)
    study = result.valid[0]

    # Corrupt the label with a value outside {0,1,2,4} (e.g. a stray 3, or noise).
    label_img = nib.load(str(study.label_path))
    data = np.asanyarray(label_img.dataobj).copy()
    data[0, 0, 0] = 99
    nib.save(nib.Nifti1Image(data, label_img.affine), str(study.label_path))

    report = validate_studies([study], tiny_config, sample_size=1)
    assert not report.ok
    assert any("unexpected values" in issue.problem for issue in report.issues)


def test_validate_studies_flags_nan_values(tiny_config: BraTSRawConfig) -> None:
    result = discover_studies(tiny_config)
    study = result.valid[0]

    modality_path = next(iter(study.modality_paths.values()))
    img = nib.load(str(modality_path))
    data = np.asanyarray(img.dataobj).copy()
    data[0, 0, 0] = np.nan
    nib.save(nib.Nifti1Image(data, img.affine), str(modality_path))

    report = validate_studies([study], tiny_config, sample_size=1)
    assert not report.ok
    assert any("NaN" in issue.problem for issue in report.issues)


def test_validate_studies_only_checks_the_requested_sample_size(tiny_config: BraTSRawConfig) -> None:
    result = discover_studies(tiny_config)
    report = validate_studies(list(result.valid), tiny_config, sample_size=1)
    assert report.studies_checked == 1
