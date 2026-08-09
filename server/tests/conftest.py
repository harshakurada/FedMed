"""Reuses hospital_nodes/tests/conftest.py's synthetic (non-medical) fixtures rather than
duplicating them -- Module 7's server-side tests need the same kind of small, real-shaped
BraTS-layout dataset Module 6's tests already build."""

from __future__ import annotations

from hospital_nodes.tests.conftest import (  # noqa: F401
    hospital_data_config,
    synthetic_hospital_dataset_root,
    tiny_hospital_training_config,
)
