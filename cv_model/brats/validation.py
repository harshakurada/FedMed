"""Dataset validation beyond file discovery: checks that need actual pixel data.

Discovery (`discovery.py`) already catches missing files and duplicate IDs
without touching pixel data. This module goes one level deeper on a small
*sample* of studies -- not the whole dataset, since loading every volume
just to validate it defeats the point of lazy loading -- checking things
that can only be seen once a volume is actually loaded: NaN/Inf, label
values outside what's documented in `labels.py`, and modality/label volumes
that don't share the same spatial shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import nibabel as nib
import numpy as np

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.discovery import StudyRecord


@dataclass
class StudyIssue:
    study_id: str
    problem: str


@dataclass
class ValidationReport:
    studies_checked: int
    issues: list[StudyIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def summary(self) -> str:
        if self.ok:
            return f"{self.studies_checked} stud(y/ies) checked, no issues found."
        lines = [f"{self.studies_checked} stud(y/ies) checked, {len(self.issues)} issue(s):"]
        lines += [f"  - {issue.study_id}: {issue.problem}" for issue in self.issues]
        return "\n".join(lines)


def validate_studies(
    studies: list[StudyRecord],
    config: BraTSRawConfig = None,
    sample_size: int | None = None,
) -> ValidationReport:
    """Load `sample_size` studies' modality + label volumes and check them.

    Every failure is recorded against its study ID and file rather than
    raising immediately, so one bad study doesn't hide problems in the rest
    of the sample.
    """
    config = config or BraTSRawConfig()
    sample_size = sample_size if sample_size is not None else config.validation_sample_size
    sample = studies[: max(0, sample_size)]

    issues: list[StudyIssue] = []
    for study in sample:
        try:
            _validate_one_study(study, config, issues)
        except Exception as exc:  # noqa: BLE001 -- record and keep validating the rest of the sample
            issues.append(StudyIssue(study.study_id, f"failed to load ({type(exc).__name__}): {exc}"))

    return ValidationReport(studies_checked=len(sample), issues=issues)


def _validate_one_study(study: StudyRecord, config: BraTSRawConfig, issues: list[StudyIssue]) -> None:
    shapes: dict[str, tuple[int, ...]] = {}

    for modality, path in study.modality_paths.items():
        img = nib.load(str(path))
        shapes[modality] = img.shape
        data = np.asanyarray(img.dataobj)
        if data.size == 0:
            issues.append(StudyIssue(study.study_id, f"modality '{modality}' ({path.name}) is empty"))
            continue
        if np.isnan(data).any():
            issues.append(StudyIssue(study.study_id, f"modality '{modality}' ({path.name}) contains NaN values"))
        if np.isinf(data).any():
            issues.append(StudyIssue(study.study_id, f"modality '{modality}' ({path.name}) contains Inf values"))

    label_img = nib.load(str(study.label_path))
    shapes["label"] = label_img.shape
    label_data = np.asanyarray(label_img.dataobj)
    if np.isnan(label_data).any() or np.isinf(label_data).any():
        issues.append(StudyIssue(study.study_id, f"label ({study.label_path.name}) contains NaN/Inf values"))

    observed_label_values = set(np.unique(label_data).astype(int).tolist())
    unexpected = observed_label_values - set(config.valid_label_values)
    if unexpected:
        issues.append(
            StudyIssue(
                study.study_id,
                f"label ({study.label_path.name}) contains unexpected values {sorted(unexpected)}, "
                f"expected only {list(config.valid_label_values)} -- see cv_model/brats/labels.py",
            )
        )

    distinct_shapes = set(shapes.values())
    if len(distinct_shapes) > 1:
        issues.append(
            StudyIssue(
                study.study_id,
                f"inconsistent spatial dimensions across modalities/label: {shapes}",
            )
        )
