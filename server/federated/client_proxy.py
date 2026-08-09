"""In-process ClientProxy: the deliberate no-network transport for Module 7.

Flower's `FedAvg` strategy (`configure_fit`/`aggregate_fit`/`configure_evaluate`/
`aggregate_evaluate`) never touches the network itself -- it only reads/writes
`FitRes`/`EvaluateRes` objects and calls `ClientProxy.fit()`/`.evaluate()`. In a live
deployment those calls go out over gRPC to a remote `flower-supernode` process; gRPC/TLS
are explicitly out of scope for this module (a later module's job, once that boundary is
approved). `InProcessClientProxy` replaces only that one networked call with a direct
Python call to a local `flwr.client.Client` (obtained the standard way, via
`NumPyClient.to_client()`) -- everything else that runs (strategy construction, client
sampling via `ClientManager`, weighted aggregation) is genuine, unmodified Flower code.
"""

from __future__ import annotations

from flwr.client import Client
from flwr.common import (
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    ReconnectIns,
)
from flwr.server.client_proxy import ClientProxy


class InProcessClientProxy(ClientProxy):
    """Wraps a local `Client` -- `fit`/`evaluate`/`get_parameters` call straight through,
    in the same process, with no serialization or socket involved."""

    def __init__(self, cid: str, client: Client) -> None:
        super().__init__(cid)
        self.node_id = hash(cid) & 0x7FFFFFFF
        self.client = client

    def get_properties(
        self, ins: GetPropertiesIns, timeout: float | None, group_id: int | None
    ) -> GetPropertiesRes:
        raise NotImplementedError("Unused by Module 7's in-process round orchestration.")

    def get_parameters(
        self, ins: GetParametersIns, timeout: float | None, group_id: int | None
    ) -> GetParametersRes:
        return self.client.get_parameters(ins)

    def fit(self, ins: FitIns, timeout: float | None, group_id: int | None) -> FitRes:
        return self.client.fit(ins)

    def evaluate(self, ins: EvaluateIns, timeout: float | None, group_id: int | None) -> EvaluateRes:
        return self.client.evaluate(ins)

    def reconnect(self, ins: ReconnectIns, timeout: float | None, group_id: int | None) -> DisconnectRes:
        raise NotImplementedError("Unused by Module 7's in-process round orchestration.")
