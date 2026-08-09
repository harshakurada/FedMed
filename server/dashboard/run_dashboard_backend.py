"""CLI entry point for the dashboard backend: starts the WebSocket server and either
runs a small, fast, REAL federated round (default) or a clearly-isolated mock event
sequence (only with `--mock` / `DASHBOARD_MOCK_MODE=true`).

    python -m server.dashboard.run_dashboard_backend
    python -m server.dashboard.run_dashboard_backend --mock

Real mode uses tiny synthetic (non-medical) fixtures -- the same pattern
Modules 7-10's own tests use -- so a developer sees genuine backend events fast, without
needing real BraTS data or a long training run. Does not run automatically as a side
effect of anything else in this project.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import tempfile
from dataclasses import replace
from pathlib import Path

from cv_model.brats.config import BraTSRawConfig
from cv_model.training.config import TrainingConfig
from server.dashboard.events import DashboardEvent, EventType
from server.dashboard.mock_events import mock_mode_enabled, run_mock_event_sequence
from server.dashboard.state import DashboardState
from server.dashboard.websocket_server import DashboardWebSocketServer
from server.federated.config import FederatedConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fedmed.dashboard.backend")


def _build_tiny_synthetic_data_config(root: Path) -> BraTSRawConfig:
    from hospital_nodes.tests.conftest import _write_synthetic_study

    for i in range(1, 13):
        _write_synthetic_study(root, f"Synth_{i:03d}", ("flair", "t1", "t1ce", "t2"), seed=i)
    return replace(
        BraTSRawConfig(), root=root, pixdim=(1.0, 1.0, 1.0), patch_size=(8, 8, 4), val_fraction=0.25,
        seed=0, on_incomplete_study="exclude", batch_size=1, num_workers=0,
    )


def _tiny_train_config() -> TrainingConfig:
    return replace(TrainingConfig(), unet_channels=(4, 8), unet_strides=(2,), unet_num_res_units=1, device_preference="cpu")


def _run_real_demo_round(event_sink) -> None:
    from server.federated.experiment import run_federated_experiment

    with tempfile.TemporaryDirectory() as tmp:
        data_config = _build_tiny_synthetic_data_config(Path(tmp))
        federated_config = FederatedConfig(
            num_rounds=2, min_available_clients=3, min_fit_clients=3, min_evaluate_clients=3,
            fraction_fit=1.0, fraction_evaluate=0.0, local_epochs=1, local_val_fraction=0.0, seed=0,
            checkpoint_dir=Path(tmp) / "federated_run",
        )
        run_federated_experiment(
            federated_config, experiment_name="dashboard_demo", data_config=data_config,
            base_train_config=_tiny_train_config(), event_sink=event_sink,
        )


async def main_async(use_mock: bool) -> None:
    state = DashboardState()
    server = DashboardWebSocketServer(state)
    await server.start()
    server.emit(DashboardEvent(event_type=EventType.SYSTEM_READY, source="server"))

    logger.info("dashboard backend ready (mock=%s) -- connect a dashboard to ws://%s:%d", use_mock, server.host, server.port)

    if use_mock:
        await asyncio.to_thread(run_mock_event_sequence, server)
    else:
        await asyncio.to_thread(_run_real_demo_round, server)

    logger.info("demo round finished -- server keeps running so late-connecting dashboards still see the snapshot")
    await asyncio.Event().wait()  # keep serving the final snapshot until interrupted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", action="store_true", help="Use the isolated mock event generator instead of a real round.")
    args = parser.parse_args()

    use_mock = args.mock or mock_mode_enabled()
    try:
        asyncio.run(main_async(use_mock))
    except KeyboardInterrupt:
        logger.info("shutting down")


if __name__ == "__main__":
    main()
