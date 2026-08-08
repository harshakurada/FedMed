from __future__ import annotations

from pathlib import Path

from cv_model.training.results import BaselineResults


def _sample_results() -> BaselineResults:
    return BaselineResults(
        experiment_name="unit-test",
        timestamp="2026-01-01T00:00:00+00:00",
        kind="DEBUG_SANITY_TEST",
        seed=42,
        dataset_root="C:/fake/root",
        val_fraction=0.2,
        split_seed=42,
        num_train_studies=7,
        num_val_studies=2,
        model_config={"in_channels": 4, "out_channels": 3},
        training_config={"optimizer": "adam", "learning_rate": 1e-4},
        preprocessing_config={"patch_size": [128, 128, 64]},
        epochs_completed=3,
        best_epoch=2,
        stopped_early=False,
        best_val_dice=0.5,
        best_val_iou=0.35,
        best_val_loss=0.6,
        final_val_dice=0.48,
        final_val_iou=0.33,
        dice_semantics="macro mean across TC/WT/ET",
        per_class_dice={"Tumor Core": 0.4, "Whole Tumor": 0.6, "Enhancing Tumor": 0.3},
        per_class_iou={"Tumor Core": 0.3, "Whole Tumor": 0.45, "Enhancing Tumor": 0.2},
        checkpoint_reproduced_recorded_dice=True,
        training_duration_seconds=123.4,
        device="cpu",
        software_versions={"torch": "2.13.0", "monai": "1.6.0", "python": "3.14.7"},
    )


def test_results_save_load_round_trip(tmp_path: Path) -> None:
    results = _sample_results()
    path = tmp_path / "results.json"
    results.save(path)
    assert path.exists()

    loaded = BaselineResults.load(path)
    assert loaded == results


def test_results_never_contain_raw_image_data() -> None:
    # Structural guarantee, not a runtime scan: BaselineResults' fields are
    # all scalars/strings/small dicts -- there is no field capable of holding
    # a tensor or array in the first place.
    results = _sample_results()
    for value in results.__dict__.values():
        assert not hasattr(value, "shape"), "BaselineResults must never hold tensor/array data"
