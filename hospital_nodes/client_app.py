"""Flower ClientApp run by each hospital node.

Module 7: wraps `hospital_nodes.node.HospitalNode` (Module 6, no Flower import) so this
file's only job is Flower plumbing -- translating between Flower's NDArrays wire format
and `HospitalNode`'s own `get_parameters`/`set_parameters`/`fit`/`evaluate` interface. No
local-training or data-partitioning logic lives here; see `hospital_nodes/node.py` and
`hospital_nodes/simulation.py`.
"""

from __future__ import annotations

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays, Scalar

from cv_model.params import get_parameters, set_parameters
from hospital_nodes.node import HospitalNode
from hospital_nodes.simulation import create_single_hospital_node


class HospitalNodeClient(NumPyClient):
    """Flower-facing adapter around one `HospitalNode`: NDArrays<->state_dict conversion
    plus delegation to the node's own `fit`/`evaluate`, nothing else."""

    def __init__(self, node: HospitalNode) -> None:
        self.node = node

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.node.model)

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
        set_parameters(self.node.model, parameters)
        result = self.node.fit()

        # Flower's Scalar can't carry None -- val_dice/val_iou are only present when this
        # hospital has a local validation split configured (see evaluate() below).
        metrics: dict[str, Scalar] = {
            "hospital_id": result.hospital_id,
            "train_loss": result.final_train_loss,
            "train_dice": result.final_train_dice,
            "train_iou": result.final_train_iou,
        }
        if result.final_val_dice is not None:
            metrics["val_dice"] = result.final_val_dice
        if result.final_val_iou is not None:
            metrics["val_iou"] = result.final_val_iou

        return get_parameters(self.node.model), result.num_examples, metrics

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, int, dict[str, Scalar]]:
        set_parameters(self.node.model, parameters)
        result = self.node.evaluate()
        if result is None:
            raise NotImplementedError(
                f"{self.node.hospital_id} has no local validation split configured "
                "(HospitalTrainingConfig.local_val_fraction == 0.0, the default). Set "
                "'local-val-fraction' > 0 in the run config to enable per-client federated "
                "evaluation. The default path for this project is the server's centralized "
                "evaluation against Module 5's global validation set instead -- see "
                "server/federated/evaluation.py -- which does not require this method."
            )
        return result.loss, result.num_examples, {"dice": result.dice, "iou": result.iou}


def client_fn(context: Context) -> NumPyClient:
    partition_id = int(context.node_config.get("partition-id", 0))
    local_epochs = int(context.run_config.get("local-epochs", 1))
    local_val_fraction = float(context.run_config.get("local-val-fraction", 0.0))

    node = create_single_hospital_node(
        partition_id, local_epochs=local_epochs, local_val_fraction=local_val_fraction
    )
    return HospitalNodeClient(node).to_client()


app = ClientApp(client_fn=client_fn)
