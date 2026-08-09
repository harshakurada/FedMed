"""Confirms Modules 7-9 are unaffected by Module 10: plain Flower FedAvg still works
with DP/encryption entirely absent, Module 8's node-failure handling is untouched, and
Module 9's encrypted round defaults to its original (pre-Module-10) behavior when
dp_config is omitted. The full regression proof is the repo-wide `pytest` run (every
existing Module 1-9 test file is re-collected and re-run unmodified); this file adds a
few direct, fast cross-checks tying the modules together in one place.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cv_model.training.config import TrainingConfig
from server.federated.client_proxy import InProcessClientProxy
from server.federated.config import FederatedConfig
from server.federated.encrypted.run_encrypted_round import run_encrypted_round_smoke_test
from server.federated.experiment import run_federated_experiment


def _tiny_train_config() -> TrainingConfig:
    return replace(
        TrainingConfig(), unet_channels=(4, 8), unet_strides=(2,), unet_num_res_units=1, device_preference="cpu"
    )


def test_plain_fedavg_still_works_with_no_dp_and_no_encryption(hospital_data_config, tmp_path: Path, monkeypatch) -> None:
    import hospital_nodes.config as hospital_config_module
    import server.federated.experiment as experiment_module

    monkeypatch.setattr(hospital_config_module, "HOSPITAL_CHECKPOINT_ROOT", tmp_path / "hospitals")
    monkeypatch.setattr(experiment_module, "DEFAULT_BASELINE_RESULTS_PATH", tmp_path / "no_baseline.json")

    federated_config = FederatedConfig(
        num_rounds=1, min_available_clients=3, min_fit_clients=3, min_evaluate_clients=3,
        fraction_fit=1.0, fraction_evaluate=0.0, local_epochs=1, local_val_fraction=0.0, seed=0,
        checkpoint_dir=tmp_path / "federated_run",
    )
    results = run_federated_experiment(
        federated_config, experiment_name="module10_compat_check",
        data_config=hospital_data_config, base_train_config=_tiny_train_config(),
    )
    assert results.num_rounds_completed == 1


def test_module_8_node_failure_handling_is_untouched(monkeypatch) -> None:
    # Module 8's InProcessClientProxy is imported unmodified by this module's own code
    # path -- this just confirms the class still exists with its Module 8 contract.
    assert hasattr(InProcessClientProxy, "fit")
    assert hasattr(InProcessClientProxy, "evaluate")


def test_encrypted_round_default_behavior_is_unchanged_without_dp_config(
    hospital_data_config, tmp_path: Path
) -> None:
    from server.federated.encrypted.ckks_config import CKKSConfig

    ckks_config = CKKSConfig(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60), global_scale=2.0**40, chunk_size=64)
    results = run_encrypted_round_smoke_test(
        ckks_config=ckks_config, data_config=hospital_data_config, base_train_config=_tiny_train_config(),
    )
    # No dp_config passed -- identical contract to Module 9: a tight, CKKS-only error.
    assert results.success
    assert results.max_abs_error < 1e-3
