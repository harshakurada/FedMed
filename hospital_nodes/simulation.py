"""Local, no-network simulation of the 3 hospital nodes.

Builds the 3 `HospitalNode`s from a single local BraTS dataset (no Flower,
no sockets, no subprocesses) and provides a SMALL local-training sanity
check -- proving the partitioning + node + local-training wiring works.
Module 7's `hospital_nodes/client_app.py` (one hospital per process) and
`server/federated/evaluation.py` (the server's centralized eval set) both
build on the helpers below.

Usage:
    python -m hospital_nodes.simulation
"""

from __future__ import annotations

from dataclasses import replace

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.discovery import StudyRecord, discover_studies
from cv_model.brats.split import SplitResult, split_studies
from cv_model.training.config import TrainingConfig
from hospital_nodes.config import HOSPITAL_NODES, build_hospital_training_config
from hospital_nodes.model_state import states_equal
from hospital_nodes.node import HospitalNode
from hospital_nodes.partition import partition_studies, verify_partition_isolation

NUM_HOSPITALS = len(HOSPITAL_NODES)


def _discover_split_and_partition(
    data_config: BraTSRawConfig | None,
) -> tuple[list[tuple[StudyRecord, ...]], SplitResult, BraTSRawConfig]:
    """Discover studies, apply Module 3/5's existing global train/val split (preserved,
    not recomputed -- keeps federated results comparable to the centralized baseline),
    partition the TRAINING studies across the 3 hospitals, and verify isolation.

    Returns the per-hospital partitions, the `SplitResult` (so callers needing only the
    shared validation studies -- e.g. the server's centralized evaluation -- never have to
    touch a partition), and the normalized `data_config` actually used.
    """
    data_config = replace(data_config or BraTSRawConfig(), on_incomplete_study="exclude")

    discovery = discover_studies(data_config)
    split = split_studies(list(discovery.valid), val_fraction=data_config.val_fraction, seed=data_config.seed)
    print(f"Global split (unchanged from Module 3/5): {split.summary()}")

    partitions = partition_studies(list(split.train), NUM_HOSPITALS, seed=data_config.seed)
    partitions_by_name = {
        f"hospital_{chr(ord('a') + i)}": partitions[i] for i in range(NUM_HOSPITALS)
    }
    verify_partition_isolation(partitions_by_name, expected_studies=list(split.train))
    print(
        "Partition isolation verified: "
        + ", ".join(f"{name}={len(studies)}" for name, studies in partitions_by_name.items())
    )
    return partitions, split, data_config


def create_hospital_nodes(
    data_config: BraTSRawConfig = None,
    base_train_config: TrainingConfig | None = None,
    local_epochs: int = 1,
    local_val_fraction: float = 0.0,
) -> tuple[list[HospitalNode], SplitResult]:
    """Build one independent `HospitalNode` per hospital (all 3 in this process --
    Module 7's in-process federated round orchestrator uses this; a real per-process
    `ClientApp` should use `create_single_hospital_node` instead).

    Returns the nodes plus the `SplitResult` so the caller can evaluate against the same
    shared global validation set the centralized baseline used.
    """
    partitions, split, data_config = _discover_split_and_partition(data_config)
    nodes = [
        HospitalNode(
            studies=partitions[i],
            data_config=data_config,
            training_config=build_hospital_training_config(
                i,
                base_train_config=base_train_config,
                local_epochs=local_epochs,
                local_val_fraction=local_val_fraction,
            ),
        )
        for i in range(NUM_HOSPITALS)
    ]
    return nodes, split


def create_single_hospital_node(
    partition_id: int,
    data_config: BraTSRawConfig = None,
    base_train_config: TrainingConfig | None = None,
    local_epochs: int = 1,
    local_val_fraction: float = 0.0,
) -> HospitalNode:
    """Build only the requested hospital's `HospitalNode` -- what one `ClientApp` process
    needs. Still discovers/splits/partitions all 3 hospitals' studies internally (the
    partitioning is deterministic and seeded, so every process reproduces the same
    assignment independently), but constructs a model for only one of them."""
    partitions, _split, data_config = _discover_split_and_partition(data_config)
    return HospitalNode(
        studies=partitions[partition_id],
        data_config=data_config,
        training_config=build_hospital_training_config(
            partition_id,
            base_train_config=base_train_config,
            local_epochs=local_epochs,
            local_val_fraction=local_val_fraction,
        ),
    )


def get_global_validation_studies(
    data_config: BraTSRawConfig = None,
) -> tuple[tuple[StudyRecord, ...], BraTSRawConfig]:
    """Discover + apply the global split, returning only the held-out validation studies
    (never `split.train` or any hospital's partition). This is what the server's
    centralized evaluation (`server/federated/evaluation.py`) uses -- the server never
    sees a hospital's training partition."""
    data_config = replace(data_config or BraTSRawConfig(), on_incomplete_study="exclude")
    discovery = discover_studies(data_config)
    split = split_studies(list(discovery.valid), val_fraction=data_config.val_fraction, seed=data_config.seed)
    return split.val, data_config


def run_local_training_sanity_check() -> None:
    """A SMALL local-training pass on all 3 hospitals -- proves the pipeline works.
    Does NOT run a full multi-round federated experiment (there are no rounds yet --
    that's Module 7)."""
    data_config = replace(BraTSRawConfig(), patch_size=(32, 32, 16), batch_size=1)
    nodes, split = create_hospital_nodes(data_config, local_epochs=1)

    for node in nodes:
        print(f"\n{node!r}")
        before = node.get_parameters()
        result = node.fit()
        after = node.get_parameters()
        changed = not states_equal(before, after)
        print(
            f"  num_examples={result.num_examples} final_train_loss={result.final_train_loss:.4f} "
            f"final_train_dice={result.final_train_dice:.4f} weights_changed={changed}"
        )
        if not changed:
            raise RuntimeError(f"{node.hospital_id}: local training did not change model weights.")

    print("\nAll hospitals trained locally without error. Pipeline is wired correctly.")


if __name__ == "__main__":
    run_local_training_sanity_check()
