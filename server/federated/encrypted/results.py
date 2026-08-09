"""Machine-readable record for a plaintext-vs-encrypted-FedAvg comparison experiment.
One comparison record, not a second parallel history/plots/checkpoint system -- Module 7
already owns that for plaintext (`server/federated/results.py`); this mirrors its
`.save`/`.load` shape.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EncryptedExperimentResults:
    experiment_name: str
    tenseal_version: str

    poly_modulus_degree: int
    coeff_mod_bit_sizes: list[int]
    global_scale: float
    chunk_size: int
    num_clients: int
    rounds: int

    plaintext_aggregation_seconds: float
    encryption_seconds: float
    encrypted_aggregation_seconds: float
    decryption_seconds: float

    ciphertext_size_bytes: int
    public_context_size_bytes: int

    max_abs_error: float
    mean_abs_error: float
    relative_error: float
    numerical_tolerance: float
    success: bool

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "EncryptedExperimentResults":
        return cls(**json.loads(path.read_text()))
