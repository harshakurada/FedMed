"""Deterministic, safe (JSON, never pickle) model-parameter representation for
encryption.

The same iteration order is used everywhere every hospital and the global model share:
a `state_dict`'s own insertion order, guaranteed identical across hospitals since every
`HospitalNode` builds its model via the same `cv_model.model.build_unet_from_params` call
(Module 6/7) -- `state_dict()` always produces the same parameter names in the same
order for architecturally-identical models.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import torch

_TORCH_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
}


@dataclass(frozen=True)
class ParamSpec:
    """Enough metadata to reconstruct one named tensor from a flat value array --
    never any tensor data itself."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    offset: int
    length: int


class ModelMetadataError(Exception):
    """Raised when received metadata doesn't match what's expected -- wrong round,
    wrong model version, wrong parameter structure. Never silently accepted."""


def flatten_model_parameters(state_dict: dict[str, torch.Tensor]) -> tuple[np.ndarray, list[ParamSpec]]:
    """Flattens a state_dict into one 1-D float64 array (CKKS's own working precision)
    plus the `ParamSpec`s needed to reconstruct it -- deterministic, insertion-order."""
    pieces: list[np.ndarray] = []
    specs: list[ParamSpec] = []
    offset = 0
    for name, tensor in state_dict.items():
        array = tensor.detach().cpu().numpy().astype(np.float64).reshape(-1)
        dtype_name = str(tensor.dtype)
        if dtype_name.startswith("torch."):
            dtype_name = dtype_name[len("torch.") :]
        specs.append(ParamSpec(name=name, shape=tuple(tensor.shape), dtype=dtype_name, offset=offset, length=array.size))
        pieces.append(array)
        offset += array.size
    flattened = np.concatenate(pieces) if pieces else np.array([], dtype=np.float64)
    return flattened, specs


def unflatten_model_parameters(values: np.ndarray, param_specs: list[ParamSpec]) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for spec in param_specs:
        segment = values[spec.offset : spec.offset + spec.length]
        torch_dtype = _TORCH_DTYPES.get(spec.dtype, torch.float32)
        state_dict[spec.name] = torch.tensor(segment, dtype=torch_dtype).reshape(spec.shape)
    return state_dict


def serialize_model_metadata(
    param_specs: list[ParamSpec], round_id: int, model_version: str, hospital_id: str, num_examples: int
) -> bytes:
    """JSON, never pickle -- safe to deserialize from an untrusted network payload."""
    payload = {
        "round_id": round_id,
        "model_version": model_version,
        "hospital_id": hospital_id,
        "num_examples": num_examples,
        "param_specs": [asdict(spec) for spec in param_specs],
    }
    return json.dumps(payload).encode("utf-8")


def deserialize_model_metadata(data: bytes) -> dict:
    payload = json.loads(data.decode("utf-8"))
    payload["param_specs"] = [
        ParamSpec(**{**spec, "shape": tuple(spec["shape"])}) for spec in payload["param_specs"]
    ]
    return payload


def validate_model_metadata(
    metadata: dict,
    expected_param_specs: list[ParamSpec],
    expected_round_id: int,
    expected_model_version: str,
) -> None:
    """Raises `ModelMetadataError` naming exactly what mismatched. Never silently
    accepts a wrong round, wrong model version, or wrong parameter structure."""
    errors: list[str] = []
    if metadata["round_id"] != expected_round_id:
        errors.append(f"round_id mismatch: expected {expected_round_id}, got {metadata['round_id']}")
    if metadata["model_version"] != expected_model_version:
        errors.append(
            f"model_version mismatch: expected {expected_model_version!r}, got {metadata['model_version']!r}"
        )
    received_specs = metadata["param_specs"]
    if len(received_specs) != len(expected_param_specs):
        errors.append(f"parameter count mismatch: expected {len(expected_param_specs)}, got {len(received_specs)}")
    else:
        for expected, received in zip(expected_param_specs, received_specs):
            if expected.name != received.name:
                errors.append(f"parameter name mismatch: expected {expected.name!r}, got {received.name!r}")
            elif expected.shape != received.shape:
                errors.append(f"{expected.name}: shape mismatch: expected {expected.shape}, got {received.shape}")
            elif expected.dtype != received.dtype:
                errors.append(f"{expected.name}: dtype mismatch: expected {expected.dtype!r}, got {received.dtype!r}")
    if errors:
        raise ModelMetadataError("Invalid model update metadata:\n  - " + "\n  - ".join(errors))
