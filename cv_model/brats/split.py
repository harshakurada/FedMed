"""Deterministic patient-level train/validation split.

Splits whole `StudyRecord`s, never slices or sub-volumes, so no patient's
data can appear in both sets -- that would leak information between train
and validation and inflate the reported validation Dice score.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from cv_model.brats.discovery import StudyRecord


@dataclass(frozen=True)
class SplitResult:
    train: tuple[StudyRecord, ...]
    val: tuple[StudyRecord, ...]
    seed: int
    val_fraction: float

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val)

    def summary(self) -> str:
        return (
            f"{self.total} total studies -> {len(self.train)} train / {len(self.val)} val "
            f"(val_fraction={self.val_fraction}, seed={self.seed})"
        )


def split_studies(studies: list[StudyRecord], val_fraction: float, seed: int) -> SplitResult:
    """Shuffle studies deterministically (by `seed`) and cut at `val_fraction`.

    Raises if there are too few studies for `val_fraction` to place at least
    one study in each split -- silently returning an empty split is exactly
    the kind of "quiet" failure this pipeline is meant to avoid.
    """
    if not studies:
        raise ValueError("Cannot split an empty study list.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1 (exclusive), got {val_fraction}")

    ids = [s.study_id for s in studies]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate study IDs passed to split_studies -- run discovery's dedup check first.")

    shuffled = list(studies)
    random.Random(seed).shuffle(shuffled)

    num_val = round(len(shuffled) * val_fraction)
    num_val = max(1, min(num_val, len(shuffled) - 1)) if len(shuffled) >= 2 else 0
    if num_val == 0:
        raise ValueError(
            f"Only {len(shuffled)} stud(y/ies) available -- not enough to form both a "
            "train and a validation split. Provide more data or skip validation explicitly."
        )

    val = tuple(shuffled[:num_val])
    train = tuple(shuffled[num_val:])
    return SplitResult(train=train, val=val, seed=seed, val_fraction=val_fraction)
