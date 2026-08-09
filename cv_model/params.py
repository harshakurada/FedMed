"""Conversion helpers between PyTorch state_dicts and a plain list-of-ndarrays wire format.

Both the central server and every hospital node (sending/receiving local
updates each round) need the same conversion, so it lives here once instead
of being duplicated in `server/` and `hospital_nodes/`.

`NDArrays` is intentionally just `list[np.ndarray]` (the same shape as
Flower's own `flwr.common.NDArrays` alias) rather than importing that type
from `flwr` -- `cv_model/` and `hospital_nodes/`'s node-abstraction code
(Module 6) must not depend on Flower; only `hospital_nodes/client_app.py`
(the actual Flower `ClientApp`, wrapping this module) does.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

NDArrays = list[np.ndarray]


def get_parameters(model: torch.nn.Module) -> NDArrays:
    """Flatten a model's state_dict into Flower's list-of-ndarrays wire format."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: torch.nn.Module, parameters: NDArrays) -> None:
    """Load Flower ndarrays back into a model in place, in state_dict key order.

    Relies on every hospital node and the server building the model via the
    same `build_unet(config)` call, so `state_dict()` keys line up positionally.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
