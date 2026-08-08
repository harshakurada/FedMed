"""Filesystem discovery for a locally-supplied BraTS dataset.

Walks `config.root`, treating each immediate subdirectory as one
patient/study, and builds a lightweight manifest (`StudyRecord`s: an ID plus
file paths) that later modules turn into tensors. Never reads pixel data --
that only happens in `validation.py` (deep checks on a small sample) and at
actual training time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cv_model.brats.config import BraTSRawConfig


class IncompleteStudyError(Exception):
    """Raised when `on_incomplete_study="raise"` and one or more studies are missing files."""


@dataclass(frozen=True)
class StudyRecord:
    """One patient/study: an ID plus paths to its modality volumes and label -- no tensors."""

    study_id: str
    modality_paths: dict[str, Path]
    label_path: Path

    def as_manifest_dict(self) -> dict[str, str]:
        """MONAI dict-transform input: one path per modality key, plus 'label'."""
        entry = {key: str(path) for key, path in self.modality_paths.items()}
        entry["label"] = str(self.label_path)
        return entry


@dataclass(frozen=True)
class IncompleteStudy:
    """A study directory found on disk that is missing a required modality or label file."""

    study_id: str
    study_dir: Path
    missing: tuple[str, ...]


@dataclass(frozen=True)
class DiscoveryResult:
    valid: tuple[StudyRecord, ...]
    incomplete: tuple[IncompleteStudy, ...]

    def raise_if_incomplete(self) -> None:
        if not self.incomplete:
            return
        details = "; ".join(f"{s.study_id} (missing {list(s.missing)})" for s in self.incomplete)
        raise IncompleteStudyError(f"{len(self.incomplete)} incomplete stud(y/ies) found: {details}")


def _find_file(study_dir: Path, study_id: str, suffix: str, extensions: Sequence[str]) -> Path | None:
    for ext in extensions:
        candidate = study_dir / f"{study_id}_{suffix}{ext}"
        if candidate.exists():
            return candidate
    return None


def discover_studies(
    config: BraTSRawConfig,
    roots: Sequence[Path] | None = None,
) -> DiscoveryResult:
    """Build a `DiscoveryResult` from every patient directory under `roots` (default: `[config.root]`).

    `roots` accepts more than one directory so multiple BraTS releases can be
    combined; duplicate study IDs across (or within) roots raise immediately,
    since that indicates a corrupted/merged dataset rather than something
    that could be silently disambiguated.
    """
    roots = list(roots) if roots is not None else [config.root]
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(
                f"BraTS dataset root does not exist: {root}. This project does not "
                "auto-download BraTS -- set FEDMED_BRATS_ROOT to your local extracted copy."
            )

    valid: list[StudyRecord] = []
    incomplete: list[IncompleteStudy] = []
    seen_ids: set[str] = set()

    for root in roots:
        for study_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            study_id = study_dir.name
            if study_id in seen_ids:
                raise ValueError(
                    f"Duplicate study ID '{study_id}' found under {root} -- "
                    "the same patient/study cannot be discovered twice."
                )
            seen_ids.add(study_id)

            modality_paths: dict[str, Path] = {}
            missing: list[str] = []
            for modality in config.modalities:
                path = _find_file(study_dir, study_id, modality, config.file_extensions)
                if path is None:
                    missing.append(modality)
                else:
                    modality_paths[modality] = path

            label_path = _find_file(study_dir, study_id, config.label_suffix, config.file_extensions)
            if label_path is None:
                missing.append(config.label_suffix)

            if missing:
                incomplete.append(IncompleteStudy(study_id, study_dir, tuple(missing)))
                continue

            valid.append(StudyRecord(study_id=study_id, modality_paths=modality_paths, label_path=label_path))

    if not valid and not incomplete:
        raise FileNotFoundError(f"No patient/study directories found under {roots}")

    result = DiscoveryResult(valid=tuple(valid), incomplete=tuple(incomplete))
    if config.on_incomplete_study == "raise":
        result.raise_if_incomplete()
    return result
