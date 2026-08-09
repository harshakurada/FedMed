"""The federated experiment: real FedAvg over the 3 real hospital nodes, in-process.

**Why in-process, not a live network run:** Flower's Simulation Engine
(`flwr.simulation.run_simulation`) requires the `ray` package, which is not installed and
is not one of this project's approved technologies -- so this module calls Flower's own
`FedAvg` strategy directly: `configure_fit`/`aggregate_fit`/`evaluate`/
`configure_evaluate`/`aggregate_evaluate` never touch the network themselves -- they only
read/write `FitRes`/`EvaluateRes` objects. The only networked piece in a live deployment
is `ClientProxy.fit()/.evaluate()` dispatching over gRPC; `server/federated/client_proxy.py`'s
`InProcessClientProxy` replaces only that one call with a direct Python call. Everything
else that runs here -- strategy construction, client sampling, weighted aggregation -- is
genuine, unmodified Flower code. See `docs/federated_training.md` for the full writeup.

Module 8 adds real gRPC + TLS as Flower's actual live-deployment transport
(`flower-superlink`/`flower-supernode`, see `docs/secure_communication.md`) plus a small
mutual-TLS coordination service (`server/federated/grpc_service/`) -- neither replaces
this in-process orchestrator, which remains how the round loop itself is proven and
tested. Module 8 also adds this file's node-failure and stale-update handling below: a
hospital's `fit()` raising is caught (not fatal to the round) and a response naming the
wrong round is rejected before aggregation -- both real, not simulated.

Reuses Module 6's `hospital_nodes.simulation.create_hospital_nodes` (data partitioning +
node construction) and Module 1's `hospital_nodes.client_app.HospitalNodeClient`
(NumPyClient wrapper) unchanged -- no second implementation of either.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import SimpleClientManager

from cv_model.brats.config import BraTSRawConfig
from cv_model.model import build_unet_from_params
from cv_model.params import NDArrays, get_parameters, set_parameters
from cv_model.training.checkpoint import CheckpointState, save_checkpoint
from cv_model.training.config import TrainingConfig
from hospital_nodes.client_app import HospitalNodeClient
from hospital_nodes.simulation import create_hospital_nodes
from server.federated.client_proxy import InProcessClientProxy
from server.federated.config import FederatedConfig, validate_federated_config
from server.federated.evaluation import build_centralized_evaluate_fn
from server.federated.history import ClientRoundRecord, FederatedHistory, RoundRecord
from server.federated.plots import generate_federated_curves
from server.federated.results import FederatedResults, build_results, compare_to_baseline
from server.federated.strategy import build_strategy

DEFAULT_BASELINE_RESULTS_PATH = Path("./checkpoints/brats_baseline/metrics/results.json")

logger = logging.getLogger("fedmed.federated.experiment")


def _save_global_checkpoint(
    path: Path,
    round_num: int,
    ndarrays: NDArrays,
    data_config: BraTSRawConfig,
    train_config: TrainingConfig,
    metrics: dict,
) -> None:
    """Global checkpoints have no single optimizer (each hospital keeps its own local
    one, saved separately by HospitalNode) -- `optimizer_state_dict={}` reflects that
    honestly rather than pretending one exists. `hospital_id=None` marks this as a global
    (aggregated), not a per-hospital, checkpoint."""
    model = build_unet_from_params(
        data_config.in_channels,
        data_config.out_channels,
        train_config.unet_channels,
        train_config.unet_strides,
        train_config.unet_num_res_units,
        train_config.device,
    )
    set_parameters(model, ndarrays)
    state = CheckpointState(
        epoch=round_num,
        best_val_dice=float(metrics.get("dice", 0.0)),
        model_state_dict=model.state_dict(),
        optimizer_state_dict={},
        scheduler_state_dict=None,
        data_config=asdict(data_config),
        train_config=asdict(train_config),
        hospital_id=None,
        metrics=metrics,
    )
    save_checkpoint(state, path)


def run_federated_experiment(
    federated_config: FederatedConfig,
    experiment_name: str = "federated_experiment",
    data_config: BraTSRawConfig | None = None,
    base_train_config: TrainingConfig | None = None,
) -> FederatedResults:
    validate_federated_config(federated_config)

    train_config = base_train_config or TrainingConfig()
    hospitals, _split = create_hospital_nodes(
        data_config,
        base_train_config=train_config,
        local_epochs=federated_config.local_epochs,
        local_val_fraction=federated_config.local_val_fraction,
    )
    data_config = hospitals[0].data_config  # the normalized config create_hospital_nodes used

    # Initial global parameters: built the same way every hospital built its own model, so
    # shapes are guaranteed compatible -- verified explicitly below rather than trusted.
    initial_model = build_unet_from_params(
        data_config.in_channels,
        data_config.out_channels,
        train_config.unet_channels,
        train_config.unet_strides,
        train_config.unet_num_res_units,
        train_config.device,
    )
    initial_ndarrays = get_parameters(initial_model)
    initial_shapes = [arr.shape for arr in initial_ndarrays]
    for hospital in hospitals:
        hospital_shapes = [arr.shape for arr in get_parameters(hospital.model)]
        if hospital_shapes != initial_shapes:
            raise RuntimeError(
                f"{hospital.hospital_id}'s model architecture does not match the server's "
                "initial global model -- every hospital and the server must build via the "
                "same build_unet_from_params(...) call with matching config."
            )

    # Model synchronization: every hospital starts round 1 from the exact same parameters.
    for hospital in hospitals:
        set_parameters(hospital.model, initial_ndarrays)

    client_manager = SimpleClientManager()
    for hospital in hospitals:
        client = HospitalNodeClient(hospital).to_client()
        client_manager.register(InProcessClientProxy(cid=hospital.hospital_id, client=client))

    evaluate_fn = build_centralized_evaluate_fn(data_config, train_config)
    parameters = ndarrays_to_parameters(initial_ndarrays)
    strategy = build_strategy(federated_config, evaluate_fn, parameters)

    checkpoint_dir = federated_config.checkpoint_dir / "checkpoints"
    _save_global_checkpoint(checkpoint_dir / "initial_global.pt", 0, initial_ndarrays, data_config, train_config, {})

    history = FederatedHistory()
    best_global_dice = float("-inf")
    experiment_start = time.time()

    for round_num in range(1, federated_config.num_rounds + 1):
        round_start = time.time()

        fit_pairs = strategy.configure_fit(round_num, parameters, client_manager)

        # A dropped hospital (a live deployment's connection failure, simulated here by a
        # raised exception) never gets fake/substitute parameters -- it's simply excluded
        # from this round's results, exactly like Flower's own dispatch treats an
        # unreachable client. `failures` is genuine Flower failure-tracking input to
        # aggregate_fit below, not cosmetic.
        succeeded: list[tuple] = []
        failures: list[BaseException] = []
        failed_hospital_ids: list[str] = []
        for proxy, fit_ins in fit_pairs:
            try:
                succeeded.append((proxy, proxy.fit(fit_ins, timeout=None, group_id=None)))
            except Exception as exc:  # noqa: BLE001 -- a dropped hospital must never crash the round
                logger.warning("Round %d: hospital %s failed to fit: %s", round_num, proxy.cid, exc)
                failures.append(exc)
                failed_hospital_ids.append(proxy.cid)

        # Stale-update protection: a result whose echoed round (on_fit_config_fn,
        # strategy.py) doesn't match this round is a late/delayed response and must never
        # be applied to a newer global model -- excluded from aggregation entirely.
        fit_results: list[tuple] = []
        stale_hospital_ids: list[str] = []
        for proxy, fit_res in succeeded:
            response_round = fit_res.metrics.get("round")
            if response_round is not None and response_round != round_num:
                logger.warning(
                    "Round %d: rejecting stale update from %s (echoed round %s)", round_num, proxy.cid, response_round
                )
                stale_hospital_ids.append(proxy.cid)
                continue
            fit_results.append((proxy, fit_res))

        client_records = [
            ClientRoundRecord(
                hospital_id=str(fit_res.metrics.get("hospital_id", proxy.cid)),
                num_examples=fit_res.num_examples,
                train_loss=float(fit_res.metrics["train_loss"]),
                train_dice=float(fit_res.metrics["train_dice"]),
                train_iou=float(fit_res.metrics["train_iou"]),
                val_dice=float(fit_res.metrics["val_dice"]) if "val_dice" in fit_res.metrics else None,
                val_iou=float(fit_res.metrics["val_iou"]) if "val_iou" in fit_res.metrics else None,
            )
            for proxy, fit_res in fit_results
        ]

        aggregated_parameters, aggregated_fit_metrics = strategy.aggregate_fit(round_num, fit_results, failures)
        if aggregated_parameters is None:
            raise RuntimeError(
                f"Round {round_num}: FedAvg aggregation produced no parameters "
                f"({len(failed_hospital_ids)} failed, {len(stale_hospital_ids)} stale, "
                f"{len(fit_results)} usable) -- check min_fit_clients against how many "
                "hospitals are actually available."
            )
        parameters = aggregated_parameters
        aggregated_ndarrays = parameters_to_ndarrays(parameters)

        # Broadcast the aggregated global model back to every hospital so the next round's
        # local training starts from it (not from wherever each hospital's own fit() left it).
        for hospital in hospitals:
            set_parameters(hospital.model, aggregated_ndarrays)

        eval_result = strategy.evaluate(round_num, parameters)
        if eval_result is None:
            raise RuntimeError(f"Round {round_num}: centralized evaluation returned no result.")
        global_loss, global_metrics = eval_result
        global_dice = float(global_metrics["dice"])
        global_iou = float(global_metrics["iou"])

        client_dice = client_iou = None
        if federated_config.fraction_evaluate > 0.0:
            eval_pairs = strategy.configure_evaluate(round_num, parameters, client_manager)
            eval_results = [
                (proxy, proxy.evaluate(evaluate_ins, timeout=None, group_id=None)) for proxy, evaluate_ins in eval_pairs
            ]
            _, aggregated_eval_metrics = strategy.aggregate_evaluate(round_num, eval_results, [])
            client_dice = aggregated_eval_metrics.get("dice")
            client_iou = aggregated_eval_metrics.get("iou")

        record = RoundRecord(
            round=round_num,
            client_records=client_records,
            aggregated_fit_metrics=aggregated_fit_metrics,
            global_loss=global_loss,
            global_dice=global_dice,
            global_iou=global_iou,
            client_dice=client_dice,
            client_iou=client_iou,
            duration_seconds=time.time() - round_start,
            failed_hospital_ids=failed_hospital_ids,
            stale_hospital_ids=stale_hospital_ids,
        )
        history.append(record)

        _save_global_checkpoint(
            checkpoint_dir / "latest_global.pt", round_num, aggregated_ndarrays, data_config, train_config, global_metrics
        )
        if global_dice > best_global_dice:
            best_global_dice = global_dice
            _save_global_checkpoint(
                checkpoint_dir / "best_global.pt", round_num, aggregated_ndarrays, data_config, train_config, global_metrics
            )

        print(
            f"[round {round_num}/{federated_config.num_rounds}] global_loss={global_loss:.4f} "
            f"global_dice={global_dice:.4f} global_iou={global_iou:.4f} duration={record.duration_seconds:.1f}s"
        )

    history.save(federated_config.checkpoint_dir / "history" / "history.json")
    results = build_results(
        experiment_name, federated_config, data_config, train_config, history, time.time() - experiment_start
    )
    results.save(federated_config.checkpoint_dir / "metrics" / "results.json")
    generate_federated_curves(history, federated_config.checkpoint_dir / "plots")

    if DEFAULT_BASELINE_RESULTS_PATH.exists():
        comparison = compare_to_baseline(results, DEFAULT_BASELINE_RESULTS_PATH)
        print(f"Baseline comparison (centralized vs. federated): {comparison}")
    else:
        print(
            f"No centralized baseline found at {DEFAULT_BASELINE_RESULTS_PATH} -- skipping "
            "baseline comparison. Run `python -m cv_model.training.run_baseline` to produce one."
        )

    return results
