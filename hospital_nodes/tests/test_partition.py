from __future__ import annotations

import pytest

from cv_model.brats.discovery import StudyRecord
from hospital_nodes.partition import PartitionError, partition_studies, verify_partition_isolation


def _fake_studies(n: int, prefix: str = "Study") -> list[StudyRecord]:
    return [StudyRecord(study_id=f"{prefix}_{i}", modality_paths={}, label_path=None) for i in range(n)]  # type: ignore[arg-type]


def test_partition_studies_is_deterministic_for_same_seed() -> None:
    studies = _fake_studies(9)
    a = partition_studies(studies, num_partitions=3, seed=42)
    b = partition_studies(studies, num_partitions=3, seed=42)
    assert [[s.study_id for s in part] for part in a] == [[s.study_id for s in part] for part in b]


def test_partition_studies_covers_every_study_with_no_overlap() -> None:
    studies = _fake_studies(10)
    partitions = partition_studies(studies, num_partitions=3, seed=1)
    all_ids = [s.study_id for part in partitions for s in part]
    assert sorted(all_ids) == sorted(s.study_id for s in studies)
    assert len(all_ids) == len(set(all_ids))  # no duplicates -> no study in two partitions


def test_partition_studies_handles_remainder_deterministically_not_by_dropping() -> None:
    # 10 studies / 3 hospitals -> sizes 4/3/3 (round-robin), nothing discarded.
    studies = _fake_studies(10)
    partitions = partition_studies(studies, num_partitions=3, seed=1)
    sizes = sorted(len(p) for p in partitions)
    assert sizes == [3, 3, 4]
    assert sum(sizes) == 10


def test_partition_studies_rejects_duplicate_study_ids() -> None:
    studies = _fake_studies(3) + [_fake_studies(1)[0]]  # Study_0 appears twice
    with pytest.raises(PartitionError, match="Duplicate"):
        partition_studies(studies, num_partitions=3, seed=0)


def test_partition_studies_rejects_more_partitions_than_studies() -> None:
    with pytest.raises(PartitionError, match="Cannot partition"):
        partition_studies(_fake_studies(2), num_partitions=3, seed=0)


def test_verify_partition_isolation_passes_for_disjoint_partitions() -> None:
    studies = _fake_studies(9)
    partitions = partition_studies(studies, num_partitions=3, seed=0)
    by_name = {"a": partitions[0], "b": partitions[1], "c": partitions[2]}
    verify_partition_isolation(by_name, expected_studies=studies)  # should not raise


def test_verify_partition_isolation_detects_overlap() -> None:
    studies = _fake_studies(6)
    by_name = {"a": tuple(studies[:4]), "b": tuple(studies[2:])}  # studies[2:4] overlap
    with pytest.raises(PartitionError, match="overlap"):
        verify_partition_isolation(by_name)


def test_verify_partition_isolation_detects_missing_studies() -> None:
    studies = _fake_studies(6)
    by_name = {"a": tuple(studies[:2]), "b": tuple(studies[2:4])}  # studies[4:6] missing
    with pytest.raises(PartitionError, match="missing"):
        verify_partition_isolation(by_name, expected_studies=studies)


def test_verify_partition_isolation_rejects_empty_partition() -> None:
    studies = _fake_studies(3)
    by_name = {"a": tuple(studies), "b": ()}
    with pytest.raises(PartitionError, match="empty"):
        verify_partition_isolation(by_name)
