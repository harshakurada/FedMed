"""Flower ServerApp: the central aggregator every hospital node connects to.

Week 1 scope: broadcast an initial global 3D U-Net to whichever hospital
nodes have connected and confirm a full fit/aggregate/evaluate round
completes. Week 2 replaces the default `FedAvg` averaging with any
custom weighting (e.g. by per-node dataset size) once nodes hold real
BraTS partitions instead of synthetic data.
"""

from __future__ import annotations

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from cv_model.config import DEFAULT_CONFIG as DEFAULT_MODEL_CONFIG
from cv_model.model import build_unet
from cv_model.params import get_parameters
from server.config import DEFAULT_CONFIG as DEFAULT_SERVER_CONFIG


def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = int(context.run_config.get("num-server-rounds", DEFAULT_SERVER_CONFIG.num_server_rounds))

    # Build one model just to get an initial (random) set of globally-agreed
    # weights -- every hospital node overwrites its local model with these
    # before round 1, so all nodes start from the same point.
    initial_parameters = ndarrays_to_parameters(get_parameters(build_unet(DEFAULT_MODEL_CONFIG)))

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=DEFAULT_SERVER_CONFIG.num_hospital_nodes,
        min_evaluate_clients=DEFAULT_SERVER_CONFIG.num_hospital_nodes,
        min_available_clients=DEFAULT_SERVER_CONFIG.num_hospital_nodes,
        initial_parameters=initial_parameters,
    )
    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=num_rounds))


app = ServerApp(server_fn=server_fn)
