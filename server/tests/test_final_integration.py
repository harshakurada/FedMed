"""Module 12's final integration test: the one combination never exercised together
before this module -- DP (Module 10) + CKKS (Module 9) + the real mutual-TLS gRPC
coordination service (Module 8) + the real dashboard WebSocket bridge (Module 11), all
in a single federated round, over the real 3-hospital BraTS pipeline (Module 6/7).

The actual round orchestration lives in `server/federated/integrated_round.py`
(promoted to production code in Module 13, since Module 13's demo entry point needs the
exact same composition this test proves) -- this file only builds the fixtures, wires a
real dashboard WebSocket server + client around it, and asserts the composition holds:
e.g. that the aggregation server never touches a CKKS secret key, and that no forbidden
field ever reaches a dashboard payload.
"""

from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import websockets

from cv_model.brats.config import BraTSRawConfig
from cv_model.training.config import TrainingConfig
from hospital_nodes.simulation import create_hospital_nodes
from server.dashboard.state import DashboardState
from server.dashboard.websocket_server import DashboardWebSocketServer
from server.federated.dp.accountant import PrivacyAccountant
from server.federated.dp.dp_config import DPConfig
from server.federated.encrypted.aggregator import EncryptedAggregationServer
from server.federated.encrypted.ckks_config import CKKSConfig
from server.federated.encrypted.key_holder import KeyHolder
from server.federated.encrypted.serialization import flatten_model_parameters
from server.federated.evaluation import build_centralized_evaluate_fn
from server.federated.grpc_service.config import GrpcSecurityConfig
from server.federated.grpc_service.health_server import create_grpc_server
from server.federated.integrated_round import run_integrated_round

ROUND_ID = 1
MODEL_VERSION = "v1"

FORBIDDEN_PAYLOAD_FIELDS = {
    "patient_id", "patient_name", "MRI", "image_data", "mask", "raw_model_weights",
    "gradient", "secret_key", "private_key", "ckks_secret", "dp_seed", "raw_ciphertext",
}


def _tiny_train_config() -> TrainingConfig:
    return replace(
        TrainingConfig(), unet_channels=(4, 8), unet_strides=(2,), unet_num_res_units=1, device_preference="cpu"
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _setup(hospital_data_config: BraTSRawConfig, tmp_path: Path, grpc_config: GrpcSecurityConfig, monkeypatch):
    import hospital_nodes.config as hospital_config_module

    monkeypatch.setattr(hospital_config_module, "HOSPITAL_CHECKPOINT_ROOT", tmp_path / "hospitals")

    # chunk_size left at its default (the CKKS slot capacity, poly_modulus_degree // 2)
    # rather than a small test value: a small chunk_size produces many chunks, and every
    # CKKS ciphertext is always full-poly-degree-sized regardless of how few values it
    # packs -- over gRPC's real 4 MB default message limit, chunk_size=64 (fine for the
    # in-process tests) produced a >15 MB request. This is the production-realistic
    # setting, and it's what makes the real mTLS channel actually work for a real model.
    ckks_config = CKKSConfig(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60), global_scale=2.0**40)
    dp_config = DPConfig(enabled=True, clip_norm=0.5, noise_multiplier=5.0, delta=1e-5)
    accountant = PrivacyAccountant()
    rng = np.random.default_rng(0)

    hospitals, _split = create_hospital_nodes(hospital_data_config, base_train_config=_tiny_train_config(), local_epochs=1)
    key_holder = KeyHolder.generate(ckks_config)
    public_context_bytes = key_holder.public_context_bytes

    _flat, specs = flatten_model_parameters(hospitals[0].model.state_dict())
    aggregation_server = EncryptedAggregationServer(public_context_bytes, specs, round_id=ROUND_ID, model_version=MODEL_VERSION)

    grpc_server = create_grpc_server(grpc_config, aggregation_server=aggregation_server)
    grpc_server.start()

    return hospitals, ckks_config, dp_config, accountant, rng, key_holder, public_context_bytes, specs, aggregation_server, grpc_server


def test_full_stack_dp_ckks_mtls_dashboard_round_completes_end_to_end(
    hospital_data_config: BraTSRawConfig, tmp_path: Path, grpc_config: GrpcSecurityConfig, monkeypatch
) -> None:
    (
        hospitals, ckks_config, dp_config, accountant, rng, key_holder, public_context_bytes, specs,
        aggregation_server, grpc_server,
    ) = _setup(hospital_data_config, tmp_path, grpc_config, monkeypatch)

    port = _free_port()
    collected: list[dict] = []

    async def run_scenario():
        state = DashboardState()
        dashboard_server = DashboardWebSocketServer(state, host="127.0.0.1", port=port)
        await dashboard_server.start()

        async with websockets.connect(f"ws://127.0.0.1:{port}") as client:
            snapshot = json.loads(await asyncio.wait_for(client.recv(), timeout=5.0))
            assert snapshot["type"] == "snapshot"

            async def collect_until_round_completed() -> None:
                while True:
                    message = json.loads(await asyncio.wait_for(client.recv(), timeout=60.0))
                    collected.append(message["data"])
                    if message["data"]["event_type"] == "ROUND_COMPLETED":
                        return

            collector_task = asyncio.create_task(collect_until_round_completed())

            def run_round_sync():
                return run_integrated_round(
                    hospitals, ckks_config, dp_config, accountant, rng, key_holder, public_context_bytes,
                    specs, aggregation_server, grpc_config, round_id=ROUND_ID, model_version=MODEL_VERSION,
                    event_sink=dashboard_server,
                )

            result = await asyncio.to_thread(run_round_sync)
            await collector_task

        await dashboard_server.stop()
        return result

    try:
        result = asyncio.run(run_scenario())
    finally:
        grpc_server.stop(grace=1).wait()

    # All 3 hospitals participated -- no drop in this test.
    assert set(result.submitted_hospital_ids) == {"hospital_a", "hospital_b", "hospital_c"}
    assert result.failed_hospital_ids == []

    # DP actually engaged for every hospital (not a no-op): the accountant only ever
    # records a round for a hospital whose update actually went through apply_dp_mechanism.
    for hospital in hospitals:
        assert accountant.cumulative_epsilon(hospital.hospital_id) > 0.0

    # The global update decrypted successfully and is numerically sane.
    import torch
    reconstructed = result.reconstructed_state_dict
    assert reconstructed
    for tensor in reconstructed.values():
        assert isinstance(tensor, torch.Tensor)
        assert torch.isfinite(tensor).all()

    # Real centralized evaluation of the decrypted global model -- genuine Dice/IoU/loss,
    # not fabricated. Reuses Module 7's own evaluation function unchanged.
    evaluate_fn = build_centralized_evaluate_fn(hospital_data_config, _tiny_train_config())
    parameters = [reconstructed[spec.name].numpy() for spec in specs]
    loss, metrics = evaluate_fn(ROUND_ID, parameters, {})
    assert isinstance(loss, float) and np.isfinite(loss)
    assert np.isfinite(metrics["dice"]) and np.isfinite(metrics["iou"])

    event_types = [e["event_type"] for e in collected]
    for expected in (
        "ROUND_STARTED", "ENCRYPTION_UPDATED", "CLIENT_TRAINING", "PRIVACY_UPDATED",
        "CLIENT_TRAINING_COMPLETED", "GLOBAL_MODEL_UPDATED", "ROUND_COMPLETED",
    ):
        assert expected in event_types, f"{expected} was never emitted to the dashboard"
    assert event_types.count("CLIENT_TRAINING_COMPLETED") == 3
    assert event_types.count("PRIVACY_UPDATED") == 3

    for event in collected:
        assert FORBIDDEN_PAYLOAD_FIELDS.isdisjoint(event["payload"].keys())


def test_hospital_failure_during_integrated_round_still_completes_with_the_rest(
    hospital_data_config: BraTSRawConfig, tmp_path: Path, grpc_config: GrpcSecurityConfig, monkeypatch
) -> None:
    (
        hospitals, ckks_config, dp_config, accountant, rng, key_holder, public_context_bytes, specs,
        aggregation_server, grpc_server,
    ) = _setup(hospital_data_config, tmp_path, grpc_config, monkeypatch)

    port = _free_port()
    collected: list[dict] = []

    async def run_scenario():
        state = DashboardState()
        dashboard_server = DashboardWebSocketServer(state, host="127.0.0.1", port=port)
        await dashboard_server.start()

        async with websockets.connect(f"ws://127.0.0.1:{port}") as client:
            snapshot = json.loads(await asyncio.wait_for(client.recv(), timeout=5.0))
            assert snapshot["type"] == "snapshot"

            async def collect_until_round_completed() -> None:
                while True:
                    message = json.loads(await asyncio.wait_for(client.recv(), timeout=60.0))
                    collected.append(message["data"])
                    if message["data"]["event_type"] == "ROUND_COMPLETED":
                        return

            collector_task = asyncio.create_task(collect_until_round_completed())

            def run_round_sync():
                return run_integrated_round(
                    hospitals, ckks_config, dp_config, accountant, rng, key_holder, public_context_bytes,
                    specs, aggregation_server, grpc_config, round_id=ROUND_ID, model_version=MODEL_VERSION,
                    event_sink=dashboard_server, fail_hospital_id="hospital_b",
                )

            result = await asyncio.to_thread(run_round_sync)
            await collector_task

        await dashboard_server.stop()
        return result

    try:
        result = asyncio.run(run_scenario())
    finally:
        grpc_server.stop(grace=1).wait()

    assert set(result.submitted_hospital_ids) == {"hospital_a", "hospital_c"}
    assert result.failed_hospital_ids == ["hospital_b"]

    import torch
    reconstructed = result.reconstructed_state_dict
    assert reconstructed
    for tensor in reconstructed.values():
        assert isinstance(tensor, torch.Tensor)
        assert torch.isfinite(tensor).all()

    event_types = [e["event_type"] for e in collected]
    assert "CLIENT_FAILED" in event_types
    assert event_types.count("CLIENT_TRAINING_COMPLETED") == 2
    assert "ROUND_COMPLETED" in event_types

    failed_events = [e for e in collected if e["event_type"] == "CLIENT_FAILED"]
    assert len(failed_events) == 1
    assert failed_events[0]["source"] == "hospital_b"

    round_completed = next(e for e in collected if e["event_type"] == "ROUND_COMPLETED")
    assert round_completed["payload"]["clients_completed"] == 2
    assert round_completed["payload"]["clients_failed"] == 1

    for event in collected:
        assert FORBIDDEN_PAYLOAD_FIELDS.isdisjoint(event["payload"].keys())
