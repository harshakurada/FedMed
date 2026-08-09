from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cv_model.model import build_unet_from_params
from hospital_nodes.model_state import (
    clone_model_state,
    get_model_state,
    load_model_state,
    load_model_state_from_disk,
    save_model_state,
    states_equal,
)


def _tiny_model() -> torch.nn.Module:
    return build_unet_from_params(4, 3, channels=(4, 8), strides=(2,), num_res_units=1)


def test_get_model_state_returns_a_copy_not_a_live_reference() -> None:
    model = _tiny_model()
    state = get_model_state(model)
    key = next(iter(state))
    state[key] += 1.0  # mutate the returned copy
    assert not torch.equal(state[key], model.state_dict()[key]), "mutating the returned state must not mutate the model"


def test_load_model_state_transfers_weights_between_independent_models() -> None:
    model_a = _tiny_model()
    model_b = _tiny_model()
    assert not states_equal(get_model_state(model_a), get_model_state(model_b)), "freshly-initialized models should differ"

    load_model_state(model_b, get_model_state(model_a))
    assert states_equal(get_model_state(model_a), get_model_state(model_b))


def test_load_model_state_deep_copies_so_later_training_does_not_mutate_the_source() -> None:
    model_a = _tiny_model()
    model_b = _tiny_model()
    source_state = get_model_state(model_a)

    load_model_state(model_b, source_state)
    with torch.no_grad():
        for p in model_b.parameters():
            p.add_(1.0)  # simulate a training step mutating model_b's weights

    assert states_equal(source_state, get_model_state(model_a)), "model_a's original state must be untouched"
    assert not states_equal(get_model_state(model_a), get_model_state(model_b))


def test_clone_model_state_is_independent() -> None:
    model = _tiny_model()
    state = get_model_state(model)
    clone = clone_model_state(state)
    key = next(iter(clone))
    clone[key] += 1.0
    assert not torch.equal(clone[key], state[key])


def test_save_and_load_model_state_round_trip(tmp_path: Path) -> None:
    model = _tiny_model()
    state = get_model_state(model)
    path = tmp_path / "state.pt"
    save_model_state(state, path)
    assert path.exists()

    loaded = load_model_state_from_disk(path)
    assert states_equal(state, loaded)


def test_load_model_state_from_disk_raises_clearly_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_model_state_from_disk(tmp_path / "nope.pt")
