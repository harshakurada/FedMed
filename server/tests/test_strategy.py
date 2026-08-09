"""FedAvg sanity test: proves aggregation is weighted by each client's sample count, using
Flower's own `FedAvg.aggregate_fit` (not a hand-rolled reimplementation)."""

from __future__ import annotations

import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from server.federated.client_proxy import InProcessClientProxy
from server.federated.evaluation import weighted_average


def _fit_res(value: float, num_examples: int) -> FitRes:
    return FitRes(
        status=Status(code=Code.OK, message="OK"),
        parameters=ndarrays_to_parameters([np.array([value], dtype=np.float32)]),
        num_examples=num_examples,
        metrics={},
    )


def _proxy(cid: str) -> InProcessClientProxy:
    return InProcessClientProxy(cid=cid, client=None)  # type: ignore[arg-type]


def test_fedavg_aggregation_is_weighted_by_sample_count() -> None:
    strategy = FedAvg(fit_metrics_aggregation_fn=weighted_average)
    results = [
        (_proxy("hospital_a"), _fit_res(1.0, 10)),
        (_proxy("hospital_b"), _fit_res(2.0, 10)),
        (_proxy("hospital_c"), _fit_res(3.0, 80)),
    ]
    parameters, _metrics = strategy.aggregate_fit(1, results, [])
    assert parameters is not None
    [aggregated] = parameters_to_ndarrays(parameters)
    expected = (1.0 * 10 + 2.0 * 10 + 3.0 * 80) / 100
    assert aggregated[0] == pytest.approx(expected)


def test_fedavg_aggregation_is_not_a_plain_average() -> None:
    strategy = FedAvg()
    results = [
        (_proxy("hospital_a"), _fit_res(0.0, 1)),
        (_proxy("hospital_b"), _fit_res(0.0, 1)),
        (_proxy("hospital_c"), _fit_res(100.0, 1000)),
    ]
    parameters, _metrics = strategy.aggregate_fit(1, results, [])
    assert parameters is not None
    [aggregated] = parameters_to_ndarrays(parameters)
    plain_average = (0.0 + 0.0 + 100.0) / 3
    assert aggregated[0] != pytest.approx(plain_average)


def test_weighted_average_is_the_fit_metrics_aggregation_fn_used() -> None:
    strategy = FedAvg(fit_metrics_aggregation_fn=weighted_average)
    results = [
        (_proxy("hospital_a"), _fit_res_with_metrics(10, {"train_dice": 0.5})),
        (_proxy("hospital_b"), _fit_res_with_metrics(30, {"train_dice": 0.9})),
    ]
    _parameters, metrics = strategy.aggregate_fit(1, results, [])
    assert metrics["train_dice"] == pytest.approx((10 * 0.5 + 30 * 0.9) / 40)


def _fit_res_with_metrics(num_examples: int, metrics: dict) -> FitRes:
    return FitRes(
        status=Status(code=Code.OK, message="OK"),
        parameters=ndarrays_to_parameters([np.array([0.0], dtype=np.float32)]),
        num_examples=num_examples,
        metrics=metrics,
    )
