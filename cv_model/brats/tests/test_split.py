from __future__ import annotations

import pytest

from cv_model.brats.discovery import StudyRecord
from cv_model.brats.split import split_studies


def _fake_studies(n: int) -> list[StudyRecord]:
    return [StudyRecord(study_id=f"Study_{i}", modality_paths={}, label_path=None) for i in range(n)]  # type: ignore[arg-type]


def test_split_is_deterministic_for_same_seed() -> None:
    studies = _fake_studies(10)
    a = split_studies(studies, val_fraction=0.2, seed=42)
    b = split_studies(studies, val_fraction=0.2, seed=42)
    assert [s.study_id for s in a.train] == [s.study_id for s in b.train]
    assert [s.study_id for s in a.val] == [s.study_id for s in b.val]


def test_split_is_patient_level_with_no_overlap() -> None:
    studies = _fake_studies(10)
    result = split_studies(studies, val_fraction=0.3, seed=1)
    train_ids = {s.study_id for s in result.train}
    val_ids = {s.study_id for s in result.val}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {s.study_id for s in studies}


def test_split_raises_on_empty_list() -> None:
    with pytest.raises(ValueError):
        split_studies([], val_fraction=0.2, seed=0)


def test_split_raises_when_too_few_studies_for_both_sets() -> None:
    with pytest.raises(ValueError):
        split_studies(_fake_studies(1), val_fraction=0.2, seed=0)


def test_split_different_seeds_can_produce_different_orderings() -> None:
    studies = _fake_studies(10)
    a = split_studies(studies, val_fraction=0.5, seed=1)
    b = split_studies(studies, val_fraction=0.5, seed=2)
    assert [s.study_id for s in a.train] != [s.study_id for s in b.train]
