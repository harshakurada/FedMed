"""Data loading for the BraTS/MSD Task01_BrainTumour dataset.

Wraps MONAI's `DecathlonDataset`, which transparently downloads and
extracts Task01_BrainTumour (a public, registration-free dataset derived
from BraTS -- same 4 MRI modalities and tumor labels) on first use,
verifies its checksum, and caches it under `config.data_root`.

Task01_BrainTumour ships labels only for its "training" list (the
Decathlon challenge's "test" list is unlabeled), so MONAI internally
partitions that list into train/val subsets using `val_frac`/`seed`
rather than downloading a separate validation set. Passing the same
`seed`/`val_fraction` to both the "training" and "validation" section
loads below guarantees the two splits are disjoint and reproducible.
"""

from __future__ import annotations

import random
from typing import Sequence

from monai.apps import DecathlonDataset
from monai.data import DataLoader

from cv_model.config import DEFAULT_CONFIG, BraTSConfig
from cv_model.transforms import get_train_transforms, get_val_transforms


def get_dataloaders(
    config: BraTSConfig = DEFAULT_CONFIG,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Build the training and validation DataLoaders for the centralized baseline.

    Uses `monai.data.DataLoader` (not `torch.utils.data.DataLoader`) because
    `RandCropByPosNegLabeld` yields multiple crops per volume; MONAI's loader
    supplies the `list_data_collate` collate function needed to batch that
    correctly.
    """
    train_ds = DecathlonDataset(
        root_dir=str(config.data_root),
        task="Task01_BrainTumour",
        section="training",
        transform=get_train_transforms(config),
        download=download,
        seed=config.seed,
        val_frac=config.val_fraction,
        cache_rate=0.0,  # stream from disk; raise once storage/RAM budget is known
        num_workers=config.num_workers,
    )
    val_ds = DecathlonDataset(
        root_dir=str(config.data_root),
        task="Task01_BrainTumour",
        section="validation",
        transform=get_val_transforms(config),
        download=False,  # already downloaded by the training split above
        seed=config.seed,
        val_frac=config.val_fraction,
        cache_rate=0.0,
        num_workers=config.num_workers,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    # Validation runs one full volume at a time (sliding-window inference happens
    # downstream in the training script), so batch_size is fixed at 1 here.
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def partition_indices(num_items: int, num_partitions: int, seed: int) -> Sequence[Sequence[int]]:
    """Split `num_items` dataset indices into `num_partitions` disjoint, roughly
    equal shards for simulating independent hospital silos.

    Used by the federated training modules (Week 2) to give each mock hospital
    node its own private partition of the training data -- no node ever sees
    another node's indices, mirroring the fact that real hospitals never share
    raw patient data.
    """
    indices = list(range(num_items))
    random.Random(seed).shuffle(indices)
    return [indices[i::num_partitions] for i in range(num_partitions)]
