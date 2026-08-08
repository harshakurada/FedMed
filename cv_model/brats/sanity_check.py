"""Structural sanity check for the raw BraTS pipeline, on a SMALL sample only.

Verifies discovery, deep validation, transforms, and the DataLoader all work
together end to end -- deliberately capped to a handful of studies so this
finishes in seconds even against a full BraTS release. No training happens
here; that's Module 4.

Usage:
    python -m cv_model.brats.sanity_check
"""

from __future__ import annotations

from dataclasses import replace

from cv_model.brats.config import DEFAULT_CONFIG
from cv_model.brats.dataset import build_dataset
from cv_model.brats.discovery import discover_studies
from cv_model.brats.split import split_studies
from cv_model.brats.transforms import get_train_transforms, get_val_transforms
from cv_model.brats.validation import validate_studies

SANITY_SAMPLE_SIZE = 3


def run_sanity_check() -> None:
    config = replace(DEFAULT_CONFIG, on_incomplete_study="exclude")
    print(f"Dataset root: {config.root}")

    discovery = discover_studies(config)
    print(
        f"Discovery: {len(discovery.valid)} valid stud(y/ies), "
        f"{len(discovery.incomplete)} incomplete (excluded)"
    )
    for incomplete in discovery.incomplete:
        print(f"  excluded: {incomplete.study_id} missing {list(incomplete.missing)}")
    if not discovery.valid:
        raise SystemExit("No valid studies discovered -- nothing to sanity-check.")

    sample = list(discovery.valid[:SANITY_SAMPLE_SIZE])
    print(f"Deep-validating {len(sample)} stud(y/ies) (pixel-level checks)...")
    report = validate_studies(sample, config, sample_size=len(sample))
    print(report.summary())
    if not report.ok:
        raise SystemExit("Validation found issues -- fix the dataset before continuing.")

    if len(sample) < 2:
        raise SystemExit(f"Need at least 2 studies for a train/val split sanity check, found {len(sample)}.")
    split = split_studies(sample, val_fraction=1 / len(sample), seed=config.seed)
    print(split.summary())

    print("Building train Dataset + running one study through the training transforms...")
    train_ds = build_dataset(split.train, get_train_transforms(config), config)
    train_item = train_ds[0]
    # RandCropByPosNegLabeld yields a list of `num_samples` crops per volume.
    first_crop = train_item[0] if isinstance(train_item, list) else train_item
    print(f"  train image shape={tuple(first_crop['image'].shape)} label shape={tuple(first_crop['label'].shape)}")

    print("Building val Dataset + running one study through the validation transforms...")
    val_ds = build_dataset(split.val, get_val_transforms(config), config)
    val_item = val_ds[0]
    print(f"  val   image shape={tuple(val_item['image'].shape)} label shape={tuple(val_item['label'].shape)}")

    print("Building DataLoader and fetching one training batch...")
    from monai.data import DataLoader

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    batch = next(iter(train_loader))
    print(f"  batch image shape={tuple(batch['image'].shape)} label shape={tuple(batch['label'].shape)}")

    print("\nAll sanity checks passed. Raw BraTS pipeline is wired correctly.")


if __name__ == "__main__":
    run_sanity_check()
