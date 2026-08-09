from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cv_model.brats.config import BraTSRawConfig
from cv_model.training.config import TrainingConfig
from hospital_nodes.model_state import states_equal
from server.federated.config import FederatedConfig
from server.federated.experiment import run_federated_experiment
from server.federated.history import FederatedHistory
from server.federated.results import FederatedResults


def _tiny_train_config() -> TrainingConfig:
    return replace(
        TrainingConfig(), unet_channels=(4, 8), unet_strides=(2,), unet_num_res_units=1, device_preference="cpu"
    )


def test_hospitals_synchronize_to_identical_initial_parameters(hospital_data_config: BraTSRawConfig, tmp_path: Path) -> None:
    """Critical model-synchronization check: exercises the exact sequence
    run_federated_experiment uses to build the initial global model and broadcast it --
    every hospital must start round 1 from bit-identical parameters."""
    from cv_model.model import build_unet_from_params
    from cv_model.params import get_parameters, set_parameters
    from hospital_nodes.simulation import create_hospital_nodes

    train_config = _tiny_train_config()
    hospitals, _split = create_hospital_nodes(
        hospital_data_config, base_train_config=train_config, local_epochs=1, local_val_fraction=0.0
    )
    data_config = hospitals[0].data_config
    initial_model = build_unet_from_params(
        data_config.in_channels,
        data_config.out_channels,
        train_config.unet_channels,
        train_config.unet_strides,
        train_config.unet_num_res_units,
        train_config.device,
    )
    initial_ndarrays = get_parameters(initial_model)
    for hospital in hospitals:
        set_parameters(hospital.model, initial_ndarrays)

    hospital_a, hospital_b, hospital_c = hospitals
    assert states_equal(hospital_a.get_parameters(), hospital_b.get_parameters())
    assert states_equal(hospital_b.get_parameters(), hospital_c.get_parameters())


def test_aggregated_parameters_differ_from_any_single_hospitals_own_update(
    hospital_data_config: BraTSRawConfig, tmp_path: Path
) -> None:
    """FedAvg must actually blend all 3 hospitals' updates -- the aggregated result should
    not equal any single hospital's own post-training parameters."""
    from flwr.server.strategy.aggregate import aggregate as fedavg_aggregate

    from cv_model.params import get_parameters, set_parameters
    from hospital_nodes.simulation import create_hospital_nodes

    train_config = _tiny_train_config()
    hospitals, _split = create_hospital_nodes(hospital_data_config, base_train_config=train_config, local_epochs=1)
    shared_initial = get_parameters(hospitals[0].model)
    for hospital in hospitals:
        set_parameters(hospital.model, shared_initial)

    fit_results = []
    for hospital in hospitals:
        result = hospital.fit()
        fit_results.append((get_parameters(hospital.model), result.num_examples))

    aggregated = fedavg_aggregate(fit_results)

    for hospital_ndarrays, _num_examples in fit_results:
        assert not all((a == b).all() for a, b in zip(aggregated, hospital_ndarrays))


def test_federated_experiment_smoke(hospital_data_config: BraTSRawConfig, tmp_path: Path, monkeypatch) -> None:
    """End-to-end smoke test: 2 rounds, tiny synthetic data, no real BraTS download or
    long training run required. Proves the full round loop (local train -> FedAvg
    aggregate -> broadcast -> centralized eval -> history/checkpoint/plots) works."""
    import hospital_nodes.config as hospital_config_module
    import server.federated.experiment as experiment_module

    # Redirect per-hospital local checkpoints (a Module 6 side effect of HospitalNode.fit())
    # into tmp_path too, so this test never touches the real checkpoints/hospitals/ dir.
    monkeypatch.setattr(hospital_config_module, "HOSPITAL_CHECKPOINT_ROOT", tmp_path / "hospitals")
    monkeypatch.setattr(experiment_module, "DEFAULT_BASELINE_RESULTS_PATH", tmp_path / "no_baseline_here.json")

    checkpoint_dir = tmp_path / "federated_run"
    federated_config = FederatedConfig(
        num_rounds=2,
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

    results = run_federated_experiment(
        federated_config,
        experiment_name="smoke_test",
        data_config=hospital_data_config,
        base_train_config=_tiny_train_config(),
    )

    assert isinstance(results, FederatedResults)
    assert results.num_rounds_completed == 2

    history = FederatedHistory.load(checkpoint_dir / "history" / "history.json")
    assert len(history.records) == 2
    for record in history.records:
        assert len(record.client_records) == 3
        assert isinstance(record.global_dice, float)
        # Distributed evaluation was disabled (fraction_evaluate=0.0) -- must stay None,
        # never silently mislabeled as if it were the global metric.
        assert record.client_dice is None
        assert record.client_iou is None

    checkpoints_dir = checkpoint_dir / "checkpoints"
    for name in ("initial_global.pt", "latest_global.pt", "best_global.pt"):
        assert (checkpoints_dir / name).exists(), f"missing checkpoint: {name}"

    assert (checkpoint_dir / "metrics" / "results.json").exists()
    plots_dir = checkpoint_dir / "plots"
    for plot_name in ("global_loss_curve.png", "global_dice_curve.png", "global_iou_curve.png"):
        assert (plots_dir / plot_name).exists(), f"missing plot: {plot_name}"


def test_federated_config_validation_rejects_distributed_eval_without_local_val(tmp_path: Path) -> None:
    import pytest

    from server.federated.config import FederatedConfigError, validate_federated_config

    bad_config = FederatedConfig(fraction_evaluate=0.5, local_val_fraction=0.0, checkpoint_dir=tmp_path)
    with pytest.raises(FederatedConfigError):
        validate_federated_config(bad_config)
