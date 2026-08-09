"""Module 13's single demo entry point: orchestrates real DP + real CKKS + a real
mutual-TLS gRPC channel + a real dashboard WebSocket server -- every mechanism is the
same already-tested Module 6-12 code (`apply_dp_mechanism`, `encrypt_model_update`,
`EncryptedAggregationServer`, the real `FedMedCoordination` mTLS service,
`KeyHolder.decrypt_aggregate`, `build_centralized_evaluate_fn`,
`server/federated/integrated_round.py`). No new algorithm, no new privacy/encryption/
federated-learning framework, and no second backend -- this script only starts the one
dashboard WebSocket server Module 11 already built and calls existing functions in
sequence.

    python -m server.demo.run_demo          # DEMO_MODE default (true): synthetic data
    python -m server.demo.run_demo --live    # real local BraTS2020 data (FEDMED_BRATS_ROOT)
    python -m server.demo.run_demo --demo    # force demo mode even if DEMO_MODE=false

The ONLY thing simulated for speed is the training data in DEMO MODE: small synthetic
(non-medical) MRI-shaped volumes generated in-memory, the same fixture pattern Modules
6-11's own tests already use -- never real patient data, and the dashboard's mode badge
(`SYSTEM_READY`'s `mode` payload) says so honestly. Every metric this script reports is a
real number produced by real code, on whichever data is actually configured -- never
fabricated, and never presented as a clinical result.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import socket
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from cv_model.brats.config import BraTSRawConfig
from cv_model.training.config import TrainingConfig
from hospital_nodes.simulation import create_hospital_nodes
from server.dashboard.events import DashboardEvent, EventType
from server.dashboard.state import DashboardState
from server.dashboard.websocket_server import DashboardWebSocketServer
from server.demo.demo_config import demo_mode_enabled
from server.federated.dp.accountant import PrivacyAccountant
from server.federated.dp.dp_config import DPConfig
from server.federated.encrypted.aggregator import EncryptedAggregationServer
from server.federated.encrypted.ckks_config import CKKSConfig
from server.federated.encrypted.key_holder import KeyHolder
from server.federated.encrypted.serialization import flatten_model_parameters
from server.federated.evaluation import build_centralized_evaluate_fn
from server.federated.grpc_service.certs import OpenSSLNotFoundError, generate_dev_certificates
from server.federated.grpc_service.config import GrpcSecurityConfig
from server.federated.grpc_service.health_client import check_health, create_secure_channel
from server.federated.grpc_service.health_server import create_grpc_server
from server.federated.integrated_round import run_integrated_round

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fedmed.demo")

HOSPITAL_IDS = ["hospital_a", "hospital_b", "hospital_c"]

# Kept deliberately small so the demo finishes in seconds regardless of which data
# source is configured -- not the production architecture Module 12's benchmark results
# (docs/experiments.md) used. This is disclosed in the console banner below, not hidden.
DEMO_UNET_CHANNELS = (4, 8)
DEMO_UNET_STRIDES = (2,)
DEMO_UNET_NUM_RES_UNITS = 1


def _demo_train_config() -> TrainingConfig:
    return replace(
        TrainingConfig(), unet_channels=DEMO_UNET_CHANNELS, unet_strides=DEMO_UNET_STRIDES,
        unet_num_res_units=DEMO_UNET_NUM_RES_UNITS, device_preference="cpu",
    )


def _synthetic_demo_data_config(root: Path) -> BraTSRawConfig:
    from hospital_nodes.tests.conftest import _write_synthetic_study

    for i in range(1, 13):
        _write_synthetic_study(root, f"Synth_{i:03d}", ("flair", "t1", "t1ce", "t2"), seed=i)
    return replace(
        BraTSRawConfig(), root=root, pixdim=(1.0, 1.0, 1.0), patch_size=(8, 8, 4), val_fraction=0.25,
        seed=0, on_incomplete_study="exclude", batch_size=1, num_workers=0,
    )


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def run_demo_round(dashboard_server: DashboardWebSocketServer, use_demo_data: bool, work_dir: Path) -> None:
    mode_label = "DEMO MODE" if use_demo_data else "LIVE MODE"
    dashboard_server.emit(DashboardEvent(event_type=EventType.SYSTEM_READY, source="server", payload={"mode": mode_label}))

    print(f"\n=== FEDMED {mode_label} ===")
    print(f"Model architecture: unet_channels={DEMO_UNET_CHANNELS} (lightweight, for demo speed -- "
          f"see docs/experiments.md for the production-architecture benchmark numbers).")

    if use_demo_data:
        print("Data: small synthetic (non-medical) MRI-shaped volumes, generated in-memory.")
        print("      NOT real patient data. Metrics below are real, but not a clinical result.")
        data_config = _synthetic_demo_data_config(work_dir / "synthetic_data")
    else:
        print("Data: local BraTS2020 dataset at FEDMED_BRATS_ROOT (real).")
        data_config = None  # BraTSRawConfig() default -- real, FEDMED_BRATS_ROOT-configured

    import hospital_nodes.config as hospital_config_module

    hospital_config_module.HOSPITAL_CHECKPOINT_ROOT = work_dir / "hospitals"

    ckks_config = CKKSConfig(poly_modulus_degree=8192, coeff_mod_bit_sizes=(60, 40, 40, 60), global_scale=2.0**40)
    dp_config = DPConfig(enabled=True, clip_norm=0.5, noise_multiplier=5.0, delta=1e-5)
    accountant = PrivacyAccountant()
    rng = np.random.default_rng()  # never a fixed seed -- see docs/differential_privacy.md

    hospitals, _split = create_hospital_nodes(data_config, base_train_config=_demo_train_config(), local_epochs=1)
    key_holder = KeyHolder.generate(ckks_config)
    public_context_bytes = key_holder.public_context_bytes
    _flat, specs = flatten_model_parameters(hospitals[0].model.state_dict())
    aggregation_server = EncryptedAggregationServer(public_context_bytes, specs, round_id=1, model_version="v1")

    try:
        certs = generate_dev_certificates(work_dir / "certs", HOSPITAL_IDS, force=True)
    except OpenSSLNotFoundError as exc:
        print(f"\nERROR: {exc}")
        raise

    grpc_config = GrpcSecurityConfig(
        host="127.0.0.1", port=_free_tcp_port(), certs_dir=work_dir / "certs",
        ca_cert_path=certs.ca_cert, server_cert_path=certs.server_cert, server_key_path=certs.server_key,
        timeout_seconds=5.0,
    )
    grpc_server = create_grpc_server(grpc_config, aggregation_server=aggregation_server)
    grpc_server.start()
    print(f"Real mutual-TLS gRPC coordination service listening on {grpc_config.address}")

    try:
        # Real mTLS HealthCheck per hospital -- proves TLS + certificate-bound identity
        # genuinely, before any model data moves, using the same mechanism Module 8's
        # own tests use (server/tests/test_grpc_tls.py).
        for hospital_id in HOSPITAL_IDS:
            channel = create_secure_channel(grpc_config, hospital_id)
            try:
                response = check_health(channel, hospital_id, timeout=grpc_config.timeout_seconds)
            finally:
                channel.close()
            dashboard_server.emit(DashboardEvent(event_type=EventType.CLIENT_CONNECTED, source=hospital_id))
            print(f"{hospital_id} connected over real mTLS ({response.message})")

        result = run_integrated_round(
            hospitals, ckks_config, dp_config, accountant, rng, key_holder, public_context_bytes,
            specs, aggregation_server, grpc_config, round_id=1, model_version="v1", event_sink=dashboard_server,
        )
    finally:
        grpc_server.stop(grace=1).wait()

    evaluate_fn = build_centralized_evaluate_fn(data_config, _demo_train_config())
    parameters = [result.reconstructed_state_dict[spec.name].numpy() for spec in specs]
    loss, metrics = evaluate_fn(1, parameters, {})
    dashboard_server.emit(
        DashboardEvent(
            event_type=EventType.METRICS_UPDATED, source="server", round=1,
            payload={"global_loss": float(loss), "global_dice": float(metrics["dice"]), "global_iou": float(metrics["iou"])},
        )
    )

    print(
        f"\nRound complete: {len(result.submitted_hospital_ids)} hospital(s) succeeded, "
        f"{len(result.failed_hospital_ids)} failed."
    )
    print(f"global_dice={metrics['dice']:.4f} global_iou={metrics['iou']:.4f} global_loss={loss:.4f}")
    if use_demo_data:
        print("(Real numbers, computed by the real evaluation code, on synthetic demo data -- not a clinical result.)")


async def main_async(use_demo_data: bool) -> None:
    state = DashboardState()
    server = DashboardWebSocketServer(state)
    await server.start()
    print(f"Dashboard backend ready -- connect a dashboard to ws://{server.host}:{server.port}")
    print("(In another terminal: cd dashboard && npm start)")

    with tempfile.TemporaryDirectory() as tmp:
        await asyncio.to_thread(run_demo_round, server, use_demo_data, Path(tmp))

    print("\nDemo round finished -- this server keeps running so a dashboard that connects "
          "late still sees the final snapshot. Press Ctrl+C to stop.")
    await asyncio.Event().wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="Use real local BraTS2020 data (FEDMED_BRATS_ROOT) instead of synthetic demo data.")
    parser.add_argument("--demo", action="store_true", help="Force demo (synthetic-data) mode, overriding DEMO_MODE=false.")
    args = parser.parse_args()

    if args.demo:
        use_demo_data = True
    elif args.live:
        use_demo_data = False
    else:
        use_demo_data = demo_mode_enabled()

    try:
        asyncio.run(main_async(use_demo_data))
    except KeyboardInterrupt:
        logger.info("shutting down")


if __name__ == "__main__":
    main()
