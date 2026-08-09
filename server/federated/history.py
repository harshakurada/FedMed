"""Round-by-round federated history: a plain list of records, saved as JSON. Mirrors
`cv_model.training.history.TrainingHistory`'s shape/conventions for the federated setting
-- no ML tracking platform, just a trivially inspectable JSON file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ClientRoundRecord:
    """One hospital's contribution to one round."""

    hospital_id: str
    num_examples: int
    train_loss: float
    train_dice: float
    train_iou: float
    val_dice: float | None
    val_iou: float | None


@dataclass
class RoundRecord:
    round: int
    client_records: list[ClientRoundRecord]
    aggregated_fit_metrics: dict
    # Centralized evaluation against Module 5's held-out global validation set -- the
    # number compared against the centralized baseline.
    global_loss: float
    global_dice: float
    global_iou: float
    # Sample-count-weighted average of hospitals' own local validation, only populated
    # when distributed evaluation is enabled (fraction_evaluate > 0). Deliberately never
    # named "global" -- it is a different, smaller, non-held-out set.
    client_dice: float | None
    client_iou: float | None
    duration_seconds: float


@dataclass
class FederatedHistory:
    records: list[RoundRecord] = field(default_factory=list)

    def append(self, record: RoundRecord) -> None:
        self.records.append(record)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(r) for r in self.records], indent=2))

    @classmethod
    def load(cls, path: Path) -> "FederatedHistory":
        raw = json.loads(path.read_text())
        records = [
            RoundRecord(**{**entry, "client_records": [ClientRoundRecord(**c) for c in entry["client_records"]]})
            for entry in raw
        ]
        return cls(records=records)
