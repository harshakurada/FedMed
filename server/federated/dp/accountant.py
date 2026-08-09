"""Privacy accounting -- distinct from clipping (`clipping.py`) and noise addition
(`noise.py`): this module only tracks and bounds cumulative privacy loss, implemented
from first principles in plain Python/NumPy math (no Opacus/TF-Privacy/scipy).

**Per-round epsilon** uses the classical Gaussian mechanism bound (Dwork & Roth,
*The Algorithmic Foundations of Differential Privacy*, Appendix A, Theorem A.1): for
noise std `sigma = noise_multiplier * clip_norm` and L2 sensitivity `clip_norm`, the
Gaussian mechanism is `(epsilon, delta)`-DP for
`epsilon = sqrt(2 * ln(1.25 / delta)) / noise_multiplier` (the clip norm cancels out --
noise std is already calibrated as a multiple of it). This bound is proven for
`epsilon < 1`; `compute_epsilon` reports `valid_range=False` rather than silently
returning a number outside the theorem's proven validity range.

**Cross-round composition** uses basic sequential composition (Dwork & Roth, Theorem
3.16): T rounds of the same mechanism applied to the same hospital compose to
`(T * epsilon_0, T * delta_0)`. This is a real, correct, but intentionally *conservative*
method -- an RDP/moments accountant would give tighter O(sqrt(T)) scaling instead of
O(T) -- chosen because implementing a correct moments accountant from scratch is exactly
the kind of "fake precision" this project's own instructions warn against; basic
composition is simple enough to verify by hand and never overstates the achieved privacy.

**Per-hospital tracking**: composition is about what repeated observation of *one*
hospital's updates reveals -- hospital A's and hospital B's cumulative budgets are
independent. A hospital that fails/drops a round is never charged for it (the accountant
is only ever called for a hospital that actually produced a DP update -- see
server/federated/encrypted/run_encrypted_round.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


class PrivacyBudgetExceededError(Exception):
    """Raised when recording a round would push a hospital's cumulative epsilon past
    its configured max_epsilon (budget enforcement enabled). The system never silently
    continues past a configured budget."""


@dataclass(frozen=True)
class EpsilonResult:
    epsilon: float
    delta: float
    valid_range: bool  # False if epsilon >= 1 -- outside the classical bound's proof


def compute_epsilon(noise_multiplier: float, delta: float) -> EpsilonResult:
    if noise_multiplier <= 0.0:
        raise ValueError(f"noise_multiplier must be > 0, got {noise_multiplier}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    epsilon = math.sqrt(2.0 * math.log(1.25 / delta)) / noise_multiplier
    return EpsilonResult(epsilon=epsilon, delta=delta, valid_range=epsilon < 1.0)


@dataclass
class RoundPrivacyRecord:
    hospital_id: str
    round_id: int
    clip_norm: float
    noise_multiplier: float
    delta: float
    delta_norm_before_clip: float
    delta_norm_after_clip: float
    epsilon_this_round: float


class PrivacyAccountant:
    """Tracks cumulative (epsilon, delta) per hospital across rounds. Never resets a
    hospital's budget between rounds -- cumulative_epsilon always reflects every round
    recorded for that hospital so far."""

    def __init__(self, max_epsilon: float | None = None, budget_enforcement_enabled: bool = False) -> None:
        self.max_epsilon = max_epsilon
        self.budget_enforcement_enabled = budget_enforcement_enabled
        self._records: dict[str, list[RoundPrivacyRecord]] = {}

    def cumulative_epsilon(self, hospital_id: str) -> float:
        return sum(record.epsilon_this_round for record in self._records.get(hospital_id, []))

    def rounds_recorded(self, hospital_id: str) -> int:
        return len(self._records.get(hospital_id, []))

    def would_exceed_budget(self, hospital_id: str, additional_epsilon: float) -> bool:
        if self.max_epsilon is None:
            return False
        return (self.cumulative_epsilon(hospital_id) + additional_epsilon) > self.max_epsilon

    def record_round(
        self,
        hospital_id: str,
        round_id: int,
        clip_norm: float,
        noise_multiplier: float,
        delta: float,
        delta_norm_before_clip: float,
        delta_norm_after_clip: float,
    ) -> RoundPrivacyRecord:
        epsilon_result = compute_epsilon(noise_multiplier, delta)

        if self.budget_enforcement_enabled and self.would_exceed_budget(hospital_id, epsilon_result.epsilon):
            raise PrivacyBudgetExceededError(
                f"Recording round {round_id} for {hospital_id} would bring cumulative epsilon to "
                f"{self.cumulative_epsilon(hospital_id) + epsilon_result.epsilon:.4f}, exceeding "
                f"max_epsilon={self.max_epsilon:.4f} -- rejected."
            )

        record = RoundPrivacyRecord(
            hospital_id=hospital_id,
            round_id=round_id,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            delta=delta,
            delta_norm_before_clip=delta_norm_before_clip,
            delta_norm_after_clip=delta_norm_after_clip,
            epsilon_this_round=epsilon_result.epsilon,
        )
        self._records.setdefault(hospital_id, []).append(record)
        return record
