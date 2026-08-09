"""Deterministic patient-level partitioning of studies across simulated hospitals.

Splits the GLOBAL TRAINING studies only (Module 3/5's existing patient-level
train/val split is computed first and preserved unchanged -- see
`hospital_nodes/simulation.py`) into non-overlapping per-hospital shares.
The shared centralized validation set is never partitioned or touched here,
so federated results stay comparable to the Module 5 baseline, and no
validation study can leak into any hospital's local training data.

Uses the same deterministic round-robin slicing already established in
`cv_model/dataset.py`'s `partition_indices` (Module 1's original federated
placeholder), applied to `StudyRecord`s instead of raw indices.
"""

from __future__ import annotations

import random

from cv_model.brats.discovery import StudyRecord


class PartitionError(Exception):
    """Raised when a partition is invalid: empty, overlapping, or not covering every study."""


def partition_studies(
    studies: list[StudyRecord],
    num_partitions: int,
    seed: int,
) -> tuple[tuple[StudyRecord, ...], ...]:
    """Split `studies` into `num_partitions` disjoint, near-equal, deterministic shares.

    Round-robin over a seed-shuffled order: any remainder from uneven division
    lands on the first `len(studies) % num_partitions` partitions, one extra
    study each -- deterministic, not silently dropped.
    """
    if num_partitions <= 0:
        raise PartitionError(f"num_partitions must be > 0, got {num_partitions}")
    if len(studies) < num_partitions:
        raise PartitionError(
            f"Cannot partition {len(studies)} stud(y/ies) into {num_partitions} non-empty "
            "partitions -- every hospital must get at least one study."
        )
    study_ids = [s.study_id for s in studies]
    if len(set(study_ids)) != len(study_ids):
        raise PartitionError("Duplicate study IDs passed to partition_studies -- run discovery's dedup check first.")

    shuffled = list(studies)
    random.Random(seed).shuffle(shuffled)
    return tuple(tuple(shuffled[i::num_partitions]) for i in range(num_partitions))


def verify_partition_isolation(
    partitions: dict[str, tuple[StudyRecord, ...]],
    expected_studies: list[StudyRecord] | None = None,
) -> None:
    """Verify every partition is non-empty, no study appears in more than one partition,
    and (if `expected_studies` is given) the partitions' union covers it exactly.

    Raises `PartitionError` and does NOT allow training to proceed on any failure --
    per Module 6's data-leakage rule, this is checked before any local training runs.
    """
    id_sets: dict[str, set[str]] = {}
    for name, studies in partitions.items():
        if not studies:
            raise PartitionError(f"Partition '{name}' is empty -- every hospital needs at least one study.")
        ids = [s.study_id for s in studies]
        if len(set(ids)) != len(ids):
            raise PartitionError(f"Partition '{name}' contains duplicate study IDs: {ids}")
        id_sets[name] = set(ids)

    names = list(id_sets)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            overlap = id_sets[name_a] & id_sets[name_b]
            if overlap:
                raise PartitionError(f"Partitions '{name_a}' and '{name_b}' overlap: {sorted(overlap)}")

    if expected_studies is not None:
        expected_ids = {s.study_id for s in expected_studies}
        union = set().union(*id_sets.values())
        missing = expected_ids - union
        unexpected = union - expected_ids
        if missing:
            raise PartitionError(f"{len(missing)} stud(y/ies) missing from all partitions: {sorted(missing)}")
        if unexpected:
            raise PartitionError(f"{len(unexpected)} stud(y/ies) in partitions but not in the expected set: {sorted(unexpected)}")
