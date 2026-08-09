"""FedAvg strategy construction: the single source of truth both the live-deployment
`ServerApp` (`server/server_app.py`) and the in-process round orchestrator
(`server/federated/experiment.py`) build from, so their FedAvg configuration never drifts
apart. Uses Flower's own `FedAvg` unmodified -- no hand-rolled aggregation.
"""

from __future__ import annotations

from flwr.common import Parameters
from flwr.server.strategy import FedAvg

from server.federated.config import FederatedConfig
from server.federated.evaluation import EvaluateFn, weighted_average


def build_strategy(
    federated_config: FederatedConfig,
    evaluate_fn: EvaluateFn | None,
    initial_parameters: Parameters,
) -> FedAvg:
    return FedAvg(
        fraction_fit=federated_config.fraction_fit,
        fraction_evaluate=federated_config.fraction_evaluate,
        min_fit_clients=federated_config.min_fit_clients,
        min_evaluate_clients=federated_config.min_evaluate_clients,
        min_available_clients=federated_config.min_available_clients,
        evaluate_fn=evaluate_fn,
        # FedAvg weights each hospital's contribution by its own num_examples (see
        # flwr.server.strategy.aggregate.aggregate) -- never a plain 3-way average.
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters,
    )
