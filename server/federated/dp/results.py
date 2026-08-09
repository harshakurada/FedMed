"""Machine-readable record for a DP (optionally + CKKS) experiment. Mirrors
`server/federated/encrypted/results.py`'s shape. Privacy metrics (epsilon, delta,
clip/noise parameters) are kept in explicitly-named fields, never conflated with utility
metrics (loss/Dice/IoU) or security-mechanism fields (CKKS/TLS) -- see
docs/differential_privacy.md's "Metric labeling" section.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class DPRoundResult:
    round_id: int
    hospital_id: str

    # --- Privacy metrics ---
    clip_norm: float
    noise_multiplier: float
    delta: float
    privacy_unit: str  # always "client-level (hospital-level)" -- see __init__.py
    delta_norm_before_clip: float
    delta_norm_after_clip: float
    epsilon_this_round: float
    cumulative_epsilon: float
    budget_status: str  # "ok" or "exceeded" -- never silently exceeded

    # --- Utility metrics (never mislabeled as privacy metrics) ---
    train_loss: float
    train_dice: float
    train_iou: float

    # --- Timing (DP overhead measured separately from CKKS overhead) ---
    local_training_seconds: float
    dp_processing_seconds: float
    encryption_seconds: float | None  # None when this round didn't use CKKS

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "DPRoundResult":
        return cls(**json.loads(path.read_text()))


@dataclass
class DPExperimentResults:
    experiment_name: str
    privacy_unit: str
    rounds: list[DPRoundResult]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"experiment_name": self.experiment_name, "privacy_unit": self.privacy_unit, "rounds": [asdict(r) for r in self.rounds]}
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "DPExperimentResults":
        payload = json.loads(path.read_text())
        return cls(
            experiment_name=payload["experiment_name"],
            privacy_unit=payload["privacy_unit"],
            rounds=[DPRoundResult(**r) for r in payload["rounds"]],
        )
