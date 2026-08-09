"""Module 13: DEMO_MODE env-var parsing. `server/demo/run_demo.py`'s actual round
orchestration is covered by `server/tests/test_final_integration.py` (it calls the same
`server/federated/integrated_round.py::run_integrated_round` the demo calls) -- this file
only checks the config toggle itself, not the whole demo script (which starts a real
gRPC server and dashboard socket; that combination is exercised by
`test_final_integration.py`, and the demo script itself was verified manually -- see
docs/final_validation_report.md).
"""

from __future__ import annotations

from server.demo.demo_config import demo_mode_enabled


def test_unset_defaults_to_demo_mode_enabled(monkeypatch) -> None:
    monkeypatch.delenv("DEMO_MODE", raising=False)
    assert demo_mode_enabled() is True


def test_explicit_false_disables_demo_mode(monkeypatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "false")
    assert demo_mode_enabled() is False


def test_explicit_true_variants_enable_demo_mode(monkeypatch) -> None:
    for value in ("1", "true", "True", "yes", "on"):
        monkeypatch.setenv("DEMO_MODE", value)
        assert demo_mode_enabled() is True


def test_explicit_false_variants_disable_demo_mode(monkeypatch) -> None:
    for value in ("0", "false", "False", "no", "off"):
        monkeypatch.setenv("DEMO_MODE", value)
        assert demo_mode_enabled() is False
