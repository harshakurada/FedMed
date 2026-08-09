from __future__ import annotations

import tenseal as ts

from server.federated.encrypted.ckks_config import CKKSConfig


def test_tenseal_is_importable_and_reports_a_version() -> None:
    assert ts.__version__


def test_default_config_derives_chunk_size_from_slot_capacity() -> None:
    config = CKKSConfig(poly_modulus_degree=8192)
    assert config.slot_capacity == 4096
    assert config.chunk_size == 4096


def test_explicit_chunk_size_overrides_the_derived_default() -> None:
    config = CKKSConfig(poly_modulus_degree=8192, chunk_size=16)
    assert config.chunk_size == 16


def test_context_built_from_config_honors_its_parameters() -> None:
    config = CKKSConfig(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60), global_scale=2.0**40)
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=config.poly_modulus_degree,
        coeff_mod_bit_sizes=list(config.coeff_mod_bit_sizes),
    )
    context.global_scale = config.global_scale
    assert context.global_scale == config.global_scale
    assert context.is_private()


def test_env_override_of_poly_modulus_degree_also_changes_derived_chunk_size(monkeypatch) -> None:
    monkeypatch.setenv("FEDMED_CKKS_POLY_MODULUS_DEGREE", "4096")
    config = CKKSConfig()
    assert config.poly_modulus_degree == 4096
    assert config.chunk_size == 2048
