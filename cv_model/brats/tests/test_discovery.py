from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.discovery import IncompleteStudyError, discover_studies


def test_discover_finds_all_complete_studies(tiny_config: BraTSRawConfig) -> None:
    result = discover_studies(replace(tiny_config, on_incomplete_study="exclude"))
    assert {s.study_id for s in result.valid} == {"Synth_001", "Synth_002", "Synth_003"}


def test_discover_flags_incomplete_study_without_dropping_it_silently(tiny_config: BraTSRawConfig) -> None:
    result = discover_studies(replace(tiny_config, on_incomplete_study="exclude"))
    assert len(result.incomplete) == 1
    assert result.incomplete[0].study_id == "Synth_004"
    assert result.incomplete[0].missing == ("t2",)


def test_discover_raises_by_default_when_incomplete(tiny_config: BraTSRawConfig) -> None:
    with pytest.raises(IncompleteStudyError, match="Synth_004"):
        discover_studies(replace(tiny_config, on_incomplete_study="raise"))


def test_discover_missing_root_raises(tmp_path: Path) -> None:
    config = replace(BraTSRawConfig(), root=tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError):
        discover_studies(config)


def test_discover_duplicate_study_id_across_roots_raises(tiny_config: BraTSRawConfig) -> None:
    with pytest.raises(ValueError, match="Duplicate study ID"):
        discover_studies(tiny_config, roots=[tiny_config.root, tiny_config.root])
