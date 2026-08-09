"""Flower ServerApp: the central aggregator every hospital node connects to.

Module 7: builds its initial global model the same way every hospital builds its own
(`build_unet_from_params` + `BraTSRawConfig`/`TrainingConfig`, not the older
`cv_model.config.BraTSConfig` path Module 1 used) and shares its `FedAvg` construction
with `server/federated/experiment.py`'s in-process round orchestrator via
`server.federated.strategy.build_strategy`, so the two never drift apart. This `ServerApp`
is what a future live deployment (gRPC/TLS, a later module) would run via `flwr run`;
this module's actual proven/tested round loop is the in-process one -- see
`docs/federated_training.md`.
"""

from __future__ import annotations

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from cv_model.brats.config import DEFAULT_CONFIG as DEFAULT_DATA_CONFIG
from cv_model.model import build_unet_from_params
from cv_model.params import get_parameters
from cv_model.training.config import DEFAULT_CONFIG as DEFAULT_TRAIN_CONFIG
from server.federated.config import DEFAULT_CONFIG as DEFAULT_FEDERATED_CONFIG
from server.federated.evaluation import build_centralized_evaluate_fn
from server.federated.strategy import build_strategy


def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = int(context.run_config.get("num-server-rounds", DEFAULT_FEDERATED_CONFIG.num_rounds))

    # Build one model just to get an initial (random) set of globally-agreed weights --
    # every hospital node overwrites its local model with these before round 1, so all
    # nodes start from the same point. Same construction hospitals use, so shapes match.
    initial_model = build_unet_from_params(
        DEFAULT_DATA_CONFIG.in_channels,
        DEFAULT_DATA_CONFIG.out_channels,
        DEFAULT_TRAIN_CONFIG.unet_channels,
        DEFAULT_TRAIN_CONFIG.unet_strides,
        DEFAULT_TRAIN_CONFIG.unet_num_res_units,
        DEFAULT_TRAIN_CONFIG.device,
    )
    initial_parameters = ndarrays_to_parameters(get_parameters(initial_model))

    evaluate_fn = build_centralized_evaluate_fn(DEFAULT_DATA_CONFIG, DEFAULT_TRAIN_CONFIG)
    strategy = build_strategy(DEFAULT_FEDERATED_CONFIG, evaluate_fn, initial_parameters)
    # Module 8: bounds how long the live deployment waits on a round before giving up --
    # one unreachable hospital can't block the server indefinitely. None (the default)
    # means no deadline, matching Flower's own default.
    server_config = ServerConfig(num_rounds=num_rounds, round_timeout=DEFAULT_FEDERATED_CONFIG.round_timeout_seconds)
    return ServerAppComponents(strategy=strategy, config=server_config)


app = ServerApp(server_fn=server_fn)
