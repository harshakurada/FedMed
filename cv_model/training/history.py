"""Per-epoch training history: a plain list of records, saved as JSON.

No ML tracking platform (MLflow/W&B) -- a JSON file is enough to plot curves
later and is trivially inspectable without extra tooling.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float | None
    train_dice: float | None
    val_dice: float | None
    train_iou: float | None
    val_iou: float | None
    learning_rate: float
    duration_seconds: float


@dataclass
class TrainingHistory:
    records: list[EpochRecord] = field(default_factory=list)

    def append(self, record: EpochRecord) -> None:
        self.records.append(record)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(r) for r in self.records], indent=2))

    @classmethod
    def load(cls, path: Path) -> "TrainingHistory":
        records = [EpochRecord(**entry) for entry in json.loads(path.read_text())]
        return cls(records=records)
