from __future__ import annotations

from pathlib import Path

from cv_model.training.history import EpochRecord, TrainingHistory


def test_history_save_load_round_trip(tmp_path: Path) -> None:
    history = TrainingHistory()
    history.append(
        EpochRecord(
            epoch=1,
            train_loss=0.8,
            val_loss=0.75,
            train_dice=0.2,
            val_dice=0.25,
            train_iou=0.1,
            val_iou=0.15,
            learning_rate=1e-4,
            duration_seconds=12.3,
        )
    )
    history.append(
        EpochRecord(
            epoch=2,
            train_loss=0.7,
            val_loss=None,  # e.g. val_frequency skipped this epoch
            train_dice=0.3,
            val_dice=None,
            train_iou=0.2,
            val_iou=None,
            learning_rate=1e-4,
            duration_seconds=11.9,
        )
    )

    path = tmp_path / "history.json"
    history.save(path)
    assert path.exists()

    loaded = TrainingHistory.load(path)
    assert len(loaded.records) == 2
    assert loaded.records[0].val_loss == 0.75
    assert loaded.records[1].val_loss is None
    assert loaded.records[1].epoch == 2
