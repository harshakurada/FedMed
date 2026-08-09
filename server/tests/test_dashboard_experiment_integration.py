"""The real-time integration smoke test: a real `run_federated_experiment` (Module 7),
with one hospital deliberately failing round 1 (Module 8's resilience mechanism, same
pattern as `server/tests/test_federated_resilience.py`), wired to a real WebSocket
server, observed by a real WebSocket client -- not a mocked event source anywhere.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from dataclasses import replace
from pathlib import Path

import websockets

from cv_model.brats.config import BraTSRawConfig
from cv_model.training.config import TrainingConfig
from server.dashboard.state import DashboardState
from server.dashboard.websocket_server import DashboardWebSocketServer
from server.federated.client_proxy import InProcessClientProxy
from server.federated.config import FederatedConfig
from server.federated.experiment import run_federated_experiment


def _tiny_train_config() -> TrainingConfig:
    return replace(
        TrainingConfig(), unet_channels=(4, 8), unet_strides=(2,), unet_num_res_units=1, device_preference="cpu"
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_real_federated_round_with_a_failure_produces_the_expected_live_events(
    hospital_data_config: BraTSRawConfig, tmp_path: Path, monkeypatch
) -> None:
    import hospital_nodes.config as hospital_config_module
    import server.federated.experiment as experiment_module

    monkeypatch.setattr(hospital_config_module, "HOSPITAL_CHECKPOINT_ROOT", tmp_path / "hospitals")
    monkeypatch.setattr(experiment_module, "DEFAULT_BASELINE_RESULTS_PATH", tmp_path / "no_baseline.json")

    original_fit = InProcessClientProxy.fit
    already_failed = {"done": False}

    def flaky_fit(self, ins, timeout, group_id):
        if self.cid == "hospital_b" and not already_failed["done"]:
            already_failed["done"] = True
            raise ConnectionError("simulated dropped connection")
        return original_fit(self, ins, timeout, group_id)

    monkeypatch.setattr(InProcessClientProxy, "fit", flaky_fit)

    port = _free_port()
    collected: list[dict] = []

    async def run_scenario() -> None:
        state = DashboardState()
        server = DashboardWebSocketServer(state, host="127.0.0.1", port=port)
        await server.start()

        async with websockets.connect(f"ws://127.0.0.1:{port}") as client:
            snapshot = json.loads(await asyncio.wait_for(client.recv(), timeout=5.0))
            assert snapshot["type"] == "snapshot"

            federated_config = FederatedConfig(
                num_rounds=2, min_available_clients=2, min_fit_clients=2, min_evaluate_clients=2,
                fraction_fit=1.0, fraction_evaluate=0.0, local_epochs=1, local_val_fraction=0.0, seed=0,
                checkpoint_dir=tmp_path / "federated_run",
            )

            async def collect_until_done() -> None:
                # Two ROUND_COMPLETED events expected (2 rounds); stop once both seen.
                completed_rounds = 0
                while completed_rounds < 2:
                    message = json.loads(await asyncio.wait_for(client.recv(), timeout=30.0))
                    collected.append(message["data"])
                    if message["data"]["event_type"] == "ROUND_COMPLETED":
                        completed_rounds += 1

            collector_task = asyncio.create_task(collect_until_done())

            def run_experiment_sync() -> None:
                run_federated_experiment(
                    federated_config, experiment_name="dashboard_integration_test",
                    data_config=hospital_data_config, base_train_config=_tiny_train_config(),
                    event_sink=server,
                )

            await asyncio.to_thread(run_experiment_sync)
            await collector_task

        await server.stop()

    asyncio.run(run_scenario())

    event_types = [e["event_type"] for e in collected]
    assert "ROUND_STARTED" in event_types
    assert "CLIENT_TRAINING_COMPLETED" in event_types
    assert "CLIENT_FAILED" in event_types
    assert event_types.count("ROUND_COMPLETED") == 2

    # The failure was for hospital_b, round 1 -- confirm it's actually the source.
    failed_events = [e for e in collected if e["event_type"] == "CLIENT_FAILED"]
    assert len(failed_events) == 1
    assert failed_events[0]["source"] == "hospital_b"
    assert failed_events[0]["round"] == 1

    # Round 2's CLIENT_TRAINING_COMPLETED events should include hospital_b again --
    # i.e. it reconnected.
    round_2_completions = [
        e for e in collected if e["event_type"] == "CLIENT_TRAINING_COMPLETED" and e["round"] == 2
    ]
    assert {e["source"] for e in round_2_completions} == {"hospital_a", "hospital_b", "hospital_c"}

    # No forbidden field anywhere in any collected payload -- the actual security check
    # against real, live traffic (events.py's own construction-time check already
    # guarantees this structurally; this re-confirms it end-to-end).
    forbidden = {"patient_id", "patient_name", "MRI", "image_data", "mask", "raw_model_weights", "gradient", "secret_key", "private_key", "ckks_secret", "dp_seed", "raw_ciphertext"}
    for event in collected:
        assert forbidden.isdisjoint(event["payload"].keys())
