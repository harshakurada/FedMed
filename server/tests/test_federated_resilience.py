"""Module 8: node-failure, reconnection, and stale-update tests -- all against Module 7's
real in-process FedAvg orchestrator (server/federated/experiment.py), not a mock.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cv_model.brats.config import BraTSRawConfig
from cv_model.training.config import TrainingConfig
from server.federated.client_proxy import InProcessClientProxy
from server.federated.config import FederatedConfig
from server.federated.experiment import run_federated_experiment
from server.federated.history import FederatedHistory


def _tiny_train_config() -> TrainingConfig:
    return replace(
        TrainingConfig(), unet_channels=(4, 8), unet_strides=(2,), unet_num_res_units=1, device_preference="cpu"
    )


def _isolate_checkpoints(monkeypatch, tmp_path: Path) -> None:
    import hospital_nodes.config as hospital_config_module
    import server.federated.experiment as experiment_module

    monkeypatch.setattr(hospital_config_module, "HOSPITAL_CHECKPOINT_ROOT", tmp_path / "hospitals")
    monkeypatch.setattr(experiment_module, "DEFAULT_BASELINE_RESULTS_PATH", tmp_path / "no_baseline_here.json")


def test_one_hospital_dropping_does_not_block_the_round_and_reconnects_next_round(
    hospital_data_config: BraTSRawConfig, tmp_path: Path, monkeypatch
) -> None:
    _isolate_checkpoints(monkeypatch, tmp_path)

    original_fit = InProcessClientProxy.fit
    already_failed_once = {"hospital_b": False}

    def flaky_fit(self, ins, timeout, group_id):
        if self.cid == "hospital_b" and not already_failed_once["hospital_b"]:
            already_failed_once["hospital_b"] = True
            raise ConnectionError("simulated dropped hospital connection")
        return original_fit(self, ins, timeout, group_id)

    monkeypatch.setattr(InProcessClientProxy, "fit", flaky_fit)

    checkpoint_dir = tmp_path / "federated_run"
    federated_config = FederatedConfig(
        num_rounds=2,
        min_available_clients=2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        local_epochs=1,
        local_val_fraction=0.0,
        seed=0,
        checkpoint_dir=checkpoint_dir,
    )

    results = run_federated_experiment(
        federated_config,
        experiment_name="resilience_test",
        data_config=hospital_data_config,
        base_train_config=_tiny_train_config(),
    )
    assert results.num_rounds_completed == 2

    history = FederatedHistory.load(checkpoint_dir / "history" / "history.json")
    round1, round2 = history.records

    # Round 1: hospital_b dropped -- round still completed on the other two, no fake
    # parameters substituted for the missing hospital, and the failure is recorded.
    assert round1.failed_hospital_ids == ["hospital_b"]
    assert round1.stale_hospital_ids == []
    assert len(round1.client_records) == 2
    assert {record.hospital_id for record in round1.client_records} == {"hospital_a", "hospital_c"}
    assert isinstance(round1.global_dice, float)  # aggregation + centralized eval still completed

    # Round 2: nothing dropped -- hospital_b reconnects and fully participates again.
    assert round2.failed_hospital_ids == []
    assert len(round2.client_records) == 3
    assert {record.hospital_id for record in round2.client_records} == {"hospital_a", "hospital_b", "hospital_c"}


def test_stale_update_is_rejected_and_excluded_from_aggregation(
    hospital_data_config: BraTSRawConfig, tmp_path: Path, monkeypatch
) -> None:
    _isolate_checkpoints(monkeypatch, tmp_path)

    original_fit = InProcessClientProxy.fit

    def stale_fit(self, ins, timeout, group_id):
        fit_res = original_fit(self, ins, timeout, group_id)
        if self.cid == "hospital_c":
            # Simulate a delayed response that actually belongs to a stale, earlier round.
            fit_res.metrics["round"] = 999
        return fit_res

    monkeypatch.setattr(InProcessClientProxy, "fit", stale_fit)

    checkpoint_dir = tmp_path / "federated_run"
    federated_config = FederatedConfig(
        num_rounds=1,
        min_available_clients=3,
        min_fit_clients=3,
        min_evaluate_clients=3,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        local_epochs=1,
        local_val_fraction=0.0,
        seed=0,
        checkpoint_dir=checkpoint_dir,
    )

    run_federated_experiment(
        federated_config,
        experiment_name="stale_update_test",
        data_config=hospital_data_config,
        base_train_config=_tiny_train_config(),
    )

    history = FederatedHistory.load(checkpoint_dir / "history" / "history.json")
    [round1] = history.records

    assert round1.failed_hospital_ids == []
    assert round1.stale_hospital_ids == ["hospital_c"]
    assert len(round1.client_records) == 2
    assert {record.hospital_id for record in round1.client_records} == {"hospital_a", "hospital_b"}


def test_all_hospitals_failing_raises_clearly_instead_of_hanging_or_faking_a_result(
    hospital_data_config: BraTSRawConfig, tmp_path: Path, monkeypatch
) -> None:
    _isolate_checkpoints(monkeypatch, tmp_path)

    def always_fail(self, ins, timeout, group_id):
        raise ConnectionError("simulated total outage")

    monkeypatch.setattr(InProcessClientProxy, "fit", always_fail)

    federated_config = FederatedConfig(
        num_rounds=1,
        min_available_clients=1,
        min_fit_clients=1,
        min_evaluate_clients=1,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        local_epochs=1,
        local_val_fraction=0.0,
        seed=0,
        checkpoint_dir=tmp_path / "federated_run",
    )

    with pytest.raises(RuntimeError, match="failed"):
        run_federated_experiment(
            federated_config,
            experiment_name="total_outage_test",
            data_config=hospital_data_config,
            base_train_config=_tiny_train_config(),
        )
