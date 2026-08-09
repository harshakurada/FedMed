"""Safe model-state utilities: get/load/clone/save, all copy-on-read.

This is the framework-independent parameter representation Module 7's
Flower client will adapt (GET_PARAMETERS/SET_PARAMETERS map directly onto
`get_model_state`/`load_model_state`; `cv_model.params` provides the
ndarray-list conversion Flower's `NumPyClient` specifically needs, on top
of the state_dict these functions return). No Flower import here.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch

ModelState = dict[str, torch.Tensor]


def get_model_state(model: torch.nn.Module) -> ModelState:
    """A deep-copied state_dict -- mutating the result never mutates `model`."""
    return copy.deepcopy(model.state_dict())


def load_model_state(model: torch.nn.Module, state: ModelState) -> None:
    """Load `state` into `model` in place. Deep-copies first so later mutating
    `model`'s parameters (e.g. during training) can never mutate the caller's `state`."""
    model.load_state_dict(copy.deepcopy(state))


def clone_model_state(state: ModelState) -> ModelState:
    """An independent copy of `state` -- safe to hand to two different models."""
    return copy.deepcopy(state)


def save_model_state(state: ModelState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_model_state_from_disk(path: Path, map_location: torch.device | str = "cpu") -> ModelState:
    if not path.exists():
        raise FileNotFoundError(f"Model state file not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=True)


def states_equal(a: ModelState, b: ModelState) -> bool:
    """True if two state_dicts have identical keys and bit-identical tensor values."""
    if a.keys() != b.keys():
        return False
    return all(torch.equal(a[key], b[key]) for key in a)
