"""Flower ClientApp run by each mock hospital node.

Week 1 scope is node scaffolding only: prove 3 independently-launched
SuperNode processes (each on its own local `--clientappio-api-address` port)
can connect to the central SuperLink, receive the global 3D U-Net weights,
run a local training/eval step, and report updated weights back.

Each node trains on a synthetic patch seeded by its `partition-id` so the
3 nodes visibly hold different "local data" without needing the ~7GB BraTS
download yet. Week 2 swaps `_make_synthetic_batch` for a real partition of
`cv_model.dataset.get_dataloaders` via `cv_model.dataset.partition_indices`
-- the client's `fit`/`evaluate` shape (parameters in, parameters out) does
not change, only where the batch comes from.
"""

from __future__ import annotations

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays

from cv_model.config import DEFAULT_CONFIG, BraTSConfig
from cv_model.model import build_dice_metric, build_loss_function, build_unet
from cv_model.params import get_parameters, set_parameters
from hospital_nodes.config import get_hospital_config


def _make_synthetic_batch(config: BraTSConfig, partition_id: int, batch_size: int = 2):
    """A deterministic-per-node stand-in for a real local BraTS patch.

    Seeding on `partition_id` gives each mock hospital its own "patient
    population" so training visibly differs node-to-node, without requiring
    any dataset download for this scaffolding step.
    """
    generator = torch.Generator(device="cpu").manual_seed(1000 + partition_id)
    images = torch.randn(
        batch_size, config.in_channels, *config.patch_size, generator=generator
    ).to(config.device)
    labels = (
        torch.rand(batch_size, config.out_channels, *config.patch_size, generator=generator)
        > 0.9
    ).float().to(config.device)
    return images, labels


class HospitalNodeClient(NumPyClient):
    """Local training/eval loop for one mock hospital, wrapping the shared 3D U-Net."""

    def __init__(self, partition_id: int, config: BraTSConfig = DEFAULT_CONFIG) -> None:
        self.partition_id = partition_id
        self.hospital_name = get_hospital_config(partition_id).name
        self.config = config
        self.model = build_unet(config)

    def get_parameters(self, config: dict) -> NDArrays:
        return get_parameters(self.model)

    def fit(self, parameters: NDArrays, config: dict) -> tuple[NDArrays, int, dict]:
        set_parameters(self.model, parameters)
        images, labels = _make_synthetic_batch(self.config, self.partition_id)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        loss_fn = build_loss_function()

        optimizer.zero_grad()
        loss = loss_fn(self.model(images), labels)
        loss.backward()
        optimizer.step()

        print(f"[{self.hospital_name}] fit round loss={loss.item():.4f}")
        return get_parameters(self.model), images.shape[0], {"loss": float(loss.item())}

    def evaluate(self, parameters: NDArrays, config: dict) -> tuple[float, int, dict]:
        set_parameters(self.model, parameters)
        images, labels = _make_synthetic_batch(self.config, self.partition_id)

        self.model.eval()
        loss_fn = build_loss_function()
        dice_metric = build_dice_metric()
        with torch.no_grad():
            output = self.model(images)
            loss = loss_fn(output, labels)
            dice_metric(y_pred=(output > 0).float(), y=labels)
            mean_dice = dice_metric.aggregate().mean().item()
            dice_metric.reset()

        print(f"[{self.hospital_name}] eval loss={loss.item():.4f} dice={mean_dice:.4f}")
        return float(loss.item()), images.shape[0], {"dice": mean_dice}


def client_fn(context: Context):
    partition_id = int(context.node_config.get("partition-id", 0))
    return HospitalNodeClient(partition_id).to_client()


app = ClientApp(client_fn=client_fn)
