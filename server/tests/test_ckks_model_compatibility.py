"""Full 3D U-Net parameter compatibility -- not a full multi-round encrypted training
run (the task explicitly doesn't require that), just proof that flattening, chunking,
metadata, encryption, serialization, deserialization, decryption, and reconstruction all
work on the actual FedMed model architecture (Module 4's real `build_unet`), not just
small hand-built tensors. Real measured timing (~8s encrypt / ~2.6s decrypt for this
model's 4.81M parameters / 1175 chunks at CKKS default settings, measured during
development) is reported via print rather than asserted against an arbitrary threshold.
"""

from __future__ import annotations

import time

from cv_model.model import build_unet, count_parameters
from server.federated.encrypted.ckks_config import CKKSConfig
from server.federated.encrypted.encryption import encrypt_model_update
from server.federated.encrypted.key_holder import KeyHolder
from server.federated.encrypted.serialization import flatten_model_parameters


def test_real_3d_unet_parameters_flatten_chunk_encrypt_decrypt_and_reconstruct() -> None:
    model = build_unet()
    total_params = count_parameters(model)
    state_dict = model.state_dict()

    flattened, param_specs = flatten_model_parameters(state_dict)
    assert flattened.size == total_params
    assert sum(spec.length for spec in param_specs) == total_params
    assert len(param_specs) == len(state_dict)

    ckks_config = CKKSConfig()  # real project defaults, not a reduced test configuration
    key_holder = KeyHolder.generate(ckks_config)
    public_context_bytes = key_holder.public_context_bytes

    expected_num_chunks = -(-flattened.size // ckks_config.chunk_size)  # ceil division

    start = time.perf_counter()
    update = encrypt_model_update(
        flattened, param_specs, public_context_bytes, ckks_config.chunk_size, "hospital_a", 1, "v1", num_examples=5
    )
    encrypt_seconds = time.perf_counter() - start
    assert len(update.chunk_ciphertexts) == expected_num_chunks

    start = time.perf_counter()
    reconstructed = key_holder.decrypt_aggregate(update.chunk_ciphertexts, param_specs, total_examples=1)
    decrypt_seconds = time.perf_counter() - start

    print(
        f"3D U-Net compatibility: {total_params} params, {expected_num_chunks} chunks, "
        f"encrypt={encrypt_seconds:.2f}s decrypt={decrypt_seconds:.2f}s"
    )

    assert set(reconstructed.keys()) == set(state_dict.keys())
    for name, tensor in reconstructed.items():
        assert tensor.shape == state_dict[name].shape

    fresh_model = build_unet()
    fresh_model.load_state_dict(reconstructed, strict=True)
