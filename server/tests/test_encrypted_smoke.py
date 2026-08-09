"""The 3-hospital encrypted-FedAvg smoke test: real (tiny, fast) `HospitalNode.fit()`
results -> encrypt -> homomorphic aggregate -> decrypt -> reconstruct, compared against
plaintext FedAvg on the exact same fit results. Mirrors Module 7/8's own smoke-test
pattern (tiny synthetic BraTS-shaped fixtures, real training, real everything else).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cv_model.brats.config import BraTSRawConfig
from cv_model.training.config import TrainingConfig
from server.federated.encrypted.ckks_config import CKKSConfig
from server.federated.encrypted.results import EncryptedExperimentResults
from server.federated.encrypted.run_encrypted_round import run_encrypted_round_smoke_test


def _tiny_train_config() -> TrainingConfig:
    return replace(
        TrainingConfig(), unet_channels=(4, 8), unet_strides=(2,), unet_num_res_units=1, device_preference="cpu"
    )


def test_three_hospital_encrypted_round_smoke_test(hospital_data_config: BraTSRawConfig, tmp_path: Path) -> None:
    ckks_config = CKKSConfig(
        poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60), global_scale=2.0**40, chunk_size=64
    )

    results = run_encrypted_round_smoke_test(
        ckks_config=ckks_config,
        data_config=hospital_data_config,
        base_train_config=_tiny_train_config(),
        experiment_name="pytest_encrypted_smoke",
    )

    print(
        f"encrypted smoke test: max_abs_error={results.max_abs_error:.3e} "
        f"mean_abs_error={results.mean_abs_error:.3e} relative_error={results.relative_error:.3e} "
        f"encryption={results.encryption_seconds:.3f}s aggregation={results.encrypted_aggregation_seconds:.3f}s "
        f"decryption={results.decryption_seconds:.3f}s ciphertext_size_bytes={results.ciphertext_size_bytes}"
    )

    assert isinstance(results, EncryptedExperimentResults)
    assert results.num_clients == 3
    assert results.rounds == 1
    assert results.success
    assert results.max_abs_error < ckks_config.numerical_tolerance
    assert results.ciphertext_size_bytes > 0
    assert results.public_context_size_bytes > 0

    # Results are a real, machine-readable JSON record -- save/load round-trips.
    results_path = tmp_path / "results.json"
    results.save(results_path)
    reloaded = EncryptedExperimentResults.load(results_path)
    assert reloaded == results
