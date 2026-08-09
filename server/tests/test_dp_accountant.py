from __future__ import annotations

import pytest

from server.federated.dp.accountant import (
    PrivacyAccountant,
    PrivacyBudgetExceededError,
    compute_epsilon,
)


def _record(accountant: PrivacyAccountant, hospital_id: str, round_id: int) -> None:
    accountant.record_round(
        hospital_id, round_id, clip_norm=1.0, noise_multiplier=5.0, delta=1e-5,
        delta_norm_before_clip=2.0, delta_norm_after_clip=1.0,
    )


def test_epsilon_after_one_round_is_positive() -> None:
    accountant = PrivacyAccountant()
    _record(accountant, "hospital_a", 1)
    assert accountant.cumulative_epsilon("hospital_a") > 0.0


def test_cumulative_epsilon_increases_and_is_never_reset_across_rounds() -> None:
    accountant = PrivacyAccountant()
    epsilons = []
    for round_id in range(1, 6):
        _record(accountant, "hospital_a", round_id)
        epsilons.append(accountant.cumulative_epsilon("hospital_a"))
    assert epsilons == sorted(epsilons)
    assert all(b > a for a, b in zip(epsilons, epsilons[1:]))


def test_basic_composition_matches_t_times_per_round_epsilon_exactly() -> None:
    """The implemented method IS basic sequential composition -- verify the exact
    T * epsilon_0 relationship, not just "increases"."""
    accountant = PrivacyAccountant()
    per_round = compute_epsilon(noise_multiplier=5.0, delta=1e-5).epsilon
    for round_id in range(1, 5):
        _record(accountant, "hospital_a", round_id)
    assert accountant.cumulative_epsilon("hospital_a") == pytest.approx(4 * per_round)


def test_hospitals_are_tracked_independently() -> None:
    accountant = PrivacyAccountant()
    _record(accountant, "hospital_a", 1)
    _record(accountant, "hospital_a", 2)
    _record(accountant, "hospital_b", 1)
    assert accountant.cumulative_epsilon("hospital_b") < accountant.cumulative_epsilon("hospital_a")
    assert accountant.rounds_recorded("hospital_b") == 1
    assert accountant.rounds_recorded("hospital_a") == 2


def test_a_hospital_never_recorded_has_zero_budget_consumed() -> None:
    accountant = PrivacyAccountant()
    _record(accountant, "hospital_a", 1)
    # hospital_b never participated (e.g. dropped the round, Module 8's resilience path)
    # -- must not be charged.
    assert accountant.cumulative_epsilon("hospital_b") == 0.0
    assert accountant.rounds_recorded("hospital_b") == 0


def test_budget_enforcement_raises_once_max_epsilon_would_be_exceeded() -> None:
    per_round = compute_epsilon(noise_multiplier=5.0, delta=1e-5).epsilon
    accountant = PrivacyAccountant(max_epsilon=per_round * 2.5, budget_enforcement_enabled=True)
    _record(accountant, "hospital_a", 1)
    _record(accountant, "hospital_a", 2)
    with pytest.raises(PrivacyBudgetExceededError):
        _record(accountant, "hospital_a", 3)


def test_budget_enforcement_disabled_never_raises() -> None:
    per_round = compute_epsilon(noise_multiplier=5.0, delta=1e-5).epsilon
    accountant = PrivacyAccountant(max_epsilon=per_round * 0.5, budget_enforcement_enabled=False)
    for round_id in range(1, 10):
        _record(accountant, "hospital_a", round_id)  # never raises -- enforcement is off
    assert accountant.cumulative_epsilon("hospital_a") > per_round * 0.5


def test_would_exceed_budget_check_before_recording() -> None:
    per_round = compute_epsilon(noise_multiplier=5.0, delta=1e-5).epsilon
    accountant = PrivacyAccountant(max_epsilon=per_round * 1.5, budget_enforcement_enabled=True)
    assert accountant.would_exceed_budget("hospital_a", per_round) is False
    _record(accountant, "hospital_a", 1)
    assert accountant.would_exceed_budget("hospital_a", per_round) is True


@pytest.mark.parametrize(
    "noise_multiplier,delta",
    [(0.0, 1e-5), (-1.0, 1e-5), (5.0, 0.0), (5.0, 1.0), (5.0, -0.1)],
)
def test_compute_epsilon_rejects_invalid_inputs(noise_multiplier: float, delta: float) -> None:
    with pytest.raises(ValueError):
        compute_epsilon(noise_multiplier, delta)


def test_epsilon_flags_when_outside_classical_bound_valid_range() -> None:
    # noise_multiplier=1.0 with delta=1e-5 gives epsilon > 1 -- outside the classical
    # Gaussian mechanism bound's proven validity range. Must be flagged, not hidden.
    result = compute_epsilon(noise_multiplier=1.0, delta=1e-5)
    assert result.epsilon > 1.0
    assert result.valid_range is False

    result_valid = compute_epsilon(noise_multiplier=5.0, delta=1e-5)
    assert result_valid.epsilon < 1.0
    assert result_valid.valid_range is True
