"""Confirms Modules 7-10 are unaffected by Module 11's dashboard layer -- the dashboard
observes the real system, it doesn't replace or alter it."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cv_model.training.config import TrainingConfig
from server.federated.config import FederatedConfig
from server.federated.experiment import run_federated_experiment


def _tiny_train_config() -> TrainingConfig:
    return replace(
        TrainingConfig(), unet_channels=(4, 8), unet_strides=(2,), unet_num_res_units=1, device_preference="cpu"
    )


def test_plain_fedavg_still_works_with_no_event_sink(hospital_data_config, tmp_path: Path, monkeypatch) -> None:
    import hospital_nodes.config as hospital_config_module
    import server.federated.experiment as experiment_module

    monkeypatch.setattr(hospital_config_module, "HOSPITAL_CHECKPOINT_ROOT", tmp_path / "hospitals")
    monkeypatch.setattr(experiment_module, "DEFAULT_BASELINE_RESULTS_PATH", tmp_path / "no_baseline.json")

    federated_config = FederatedConfig(
        num_rounds=1, min_available_clients=3, min_fit_clients=3, min_evaluate_clients=3,
        fraction_fit=1.0, fraction_evaluate=0.0, local_epochs=1, local_val_fraction=0.0, seed=0,
        checkpoint_dir=tmp_path / "federated_run",
    )
    # event_sink deliberately omitted -- must behave exactly as it did before Module 11.
    results = run_federated_experiment(
        federated_config, experiment_name="module11_compat_check",
        data_config=hospital_data_config, base_train_config=_tiny_train_config(),
    )
    assert results.num_rounds_completed == 1


def test_dashboard_package_does_not_import_grpc_ckks_or_dp_internals() -> None:
    # The dashboard is monitoring-only -- its own modules must not reach into Module
    # 8/9/10's actual mechanism code (TLS certs, CKKS contexts, DP noise), only into
    # server/federated/experiment.py's already-approved event hook.
    import inspect

    from server.dashboard import events, state, websocket_server

    for module in (events, state, websocket_server):
        source = inspect.getsource(module)
        assert "tenseal" not in source.lower()
        assert "ssl_server_credentials" not in source
        assert "add_gaussian_noise" not in source
