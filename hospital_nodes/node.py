"""The hospital node abstraction: local dataset, local model, local training.

No Flower, gRPC, TenSEAL, or WebSocket code anywhere in this file (or
anything it imports) -- this is deliberately the piece Module 7 will wrap
with a Flower `ClientApp`, not a piece that talks to one. Local training
reuses `cv_model.training.engine.train_one_epoch`/`validate` unchanged --
no second trainer.

RAW MEDICAL DATA NEVER LEAVES A HOSPITAL NODE: `HospitalNode` keeps its
`StudyRecord`s (file paths, not pixel data) as a private attribute. Its
public, "communication-ready" methods (`get_parameters`, `fit`, `evaluate`)
return only model weights and scalar/dict metrics -- never a `StudyRecord`,
file path, or tensor of image data.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from monai.data import DataLoader

from cv_model.brats.config import BraTSRawConfig
from cv_model.brats.dataset import build_dataset
from cv_model.brats.discovery import StudyRecord
from cv_model.brats.split import split_studies
from cv_model.brats.transforms import get_train_transforms, get_val_transforms
from cv_model.model import build_dice_metric, build_iou_metric, build_loss_function, build_unet_from_params
from cv_model.training.checkpoint import CheckpointState, save_checkpoint
from cv_model.training.engine import EpochMetrics, train_one_epoch, validate
from cv_model.training.trainer import build_optimizer, build_scheduler
from hospital_nodes.config import HospitalTrainingConfig
from hospital_nodes.model_state import ModelState, get_model_state, load_model_state


@dataclass
class LocalEpochRecord:
    """One hospital's one local epoch -- mirrors cv_model.training.history.EpochRecord's
    fields, plus the hospital identity, since these records travel independently."""

    hospital_id: str
    epoch: int
    train_loss: float
    train_dice: float
    train_iou: float
    val_loss: float | None
    val_dice: float | None
    val_iou: float | None
    learning_rate: float
    duration_seconds: float


@dataclass
class LocalTrainingResult:
    """What a future Flower `fit()` would return: no raw data, only weights + metrics."""

    hospital_id: str
    num_examples: int  # local training-set size, the FedAvg weighting signal
    epoch_records: list[LocalEpochRecord]
    final_train_loss: float
    final_train_dice: float
    final_train_iou: float
    final_val_dice: float | None
    final_val_iou: float | None


@dataclass
class LocalEvaluationResult:
    """What a future Flower `evaluate()` would return."""

    hospital_id: str
    num_examples: int
    loss: float
    dice: float
    iou: float


class HospitalNode:
    """One simulated hospital: its own data partition, its own model instance, its own
    local training loop. Constructing two `HospitalNode`s never shares a model object --
    each call to `build_unet_from_params` allocates independent parameters."""

    def __init__(
        self,
        studies: tuple[StudyRecord, ...],
        data_config: BraTSRawConfig,
        training_config: HospitalTrainingConfig,
    ) -> None:
        if not studies:
            raise ValueError(f"Hospital '{training_config.hospital.name}' was given an empty study partition.")

        self.hospital_id = f"hospital_{chr(ord('a') + training_config.hospital.partition_id)}"
        self.name = training_config.hospital.name
        self.data_config = data_config
        self.training_config = training_config

        if training_config.local_val_fraction > 0.0:
            split = split_studies(
                list(studies), val_fraction=training_config.local_val_fraction, seed=data_config.seed
            )
            self._train_studies: tuple[StudyRecord, ...] = split.train
            self._val_studies: tuple[StudyRecord, ...] = split.val
        else:
            self._train_studies = studies
            self._val_studies = ()

        train_cfg = training_config.train_config
        self.model = build_unet_from_params(
            data_config.in_channels,
            data_config.out_channels,
            train_cfg.unet_channels,
            train_cfg.unet_strides,
            train_cfg.unet_num_res_units,
            train_cfg.device,
        )

    def __repr__(self) -> str:
        # Deliberately reports counts and identity only -- never study IDs/paths here,
        # so even an incidental `print(node)` can't leak which patients a hospital holds.
        return f"HospitalNode({self.hospital_id!r}, num_local_studies={len(self._train_studies)})"

    @property
    def num_local_train_studies(self) -> int:
        return len(self._train_studies)

    @property
    def num_local_val_studies(self) -> int:
        return len(self._val_studies)

    # --- Framework-independent parameter interface (Module 7 will adapt these to Flower) ---

    def get_parameters(self) -> ModelState:
        """Conceptually Flower's GET_PARAMETERS. Returns a deep copy -- never a live
        reference into this hospital's model."""
        return get_model_state(self.model)

    def set_parameters(self, state: ModelState) -> None:
        """Conceptually Flower's SET_PARAMETERS -- e.g. loading the global model at the
        start of a federated round. Deep-copies internally; safe even if the caller
        mutates `state` afterwards."""
        load_model_state(self.model, state)

    def fit(self) -> LocalTrainingResult:
        """Conceptually Flower's FIT: local training, returning weights-ready metrics only."""
        return self.local_train()

    def evaluate(self) -> LocalEvaluationResult | None:
        """Conceptually Flower's EVALUATE. Returns None if this hospital has no local
        validation split configured (the default -- see HospitalTrainingConfig)."""
        if not self._val_studies:
            return None
        train_cfg = self.training_config.train_config
        val_loader = self._build_loader(self._val_studies, get_val_transforms(self.data_config), batch_size=1)
        metrics = validate(
            self.model, val_loader, build_loss_function(), train_cfg.device, build_dice_metric(), build_iou_metric()
        )
        return LocalEvaluationResult(
            hospital_id=self.hospital_id,
            num_examples=len(self._val_studies),
            loss=metrics.loss,
            dice=metrics.dice,
            iou=metrics.iou,
        )

    # --- Local training ---

    def _build_loader(self, studies: tuple[StudyRecord, ...], transform, batch_size: int) -> DataLoader:
        dataset = build_dataset(studies, transform, self.data_config)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=batch_size > 1,
            num_workers=self.data_config.num_workers,
            pin_memory=True,
        )

    def local_train(self) -> LocalTrainingResult:
        """Train locally for `training_config.local_epochs`, reusing
        `cv_model.training.engine.train_one_epoch`/`validate` -- no second trainer."""
        train_cfg = self.training_config.train_config
        train_loader = self._build_loader(
            self._train_studies, get_train_transforms(self.data_config), self.data_config.batch_size
        )
        val_loader = (
            self._build_loader(self._val_studies, get_val_transforms(self.data_config), batch_size=1)
            if self._val_studies
            else None
        )

        loss_fn = build_loss_function()
        dice_metric = build_dice_metric()
        iou_metric = build_iou_metric()
        optimizer = build_optimizer(self.model, train_cfg)
        scheduler = build_scheduler(optimizer, train_cfg)

        records: list[LocalEpochRecord] = []
        last_train: EpochMetrics | None = None
        last_val: EpochMetrics | None = None

        for epoch in range(1, self.training_config.local_epochs + 1):
            epoch_start = time.time()
            last_train = train_one_epoch(
                self.model, train_loader, loss_fn, optimizer, train_cfg.device, dice_metric, iou_metric, train_cfg
            )
            last_val = (
                validate(self.model, val_loader, loss_fn, train_cfg.device, dice_metric, iou_metric)
                if val_loader is not None
                else None
            )
            if scheduler is not None:
                scheduler.step()

            records.append(
                LocalEpochRecord(
                    hospital_id=self.hospital_id,
                    epoch=epoch,
                    train_loss=last_train.loss,
                    train_dice=last_train.dice,
                    train_iou=last_train.iou,
                    val_loss=last_val.loss if last_val else None,
                    val_dice=last_val.dice if last_val else None,
                    val_iou=last_val.iou if last_val else None,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    duration_seconds=time.time() - epoch_start,
                )
            )

        self._save_checkpoint(records[-1].epoch, optimizer, scheduler)

        return LocalTrainingResult(
            hospital_id=self.hospital_id,
            num_examples=len(self._train_studies),
            epoch_records=records,
            final_train_loss=last_train.loss,
            final_train_dice=last_train.dice,
            final_train_iou=last_train.iou,
            final_val_dice=last_val.dice if last_val else None,
            final_val_iou=last_val.iou if last_val else None,
        )

    def _save_checkpoint(self, epoch: int, optimizer, scheduler) -> None:
        state = CheckpointState(
            epoch=epoch,
            best_val_dice=0.0,  # local checkpoints track the latest state, not a "best" search
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
            data_config=asdict(self.data_config),
            train_config=asdict(self.training_config.train_config),
            hospital_id=self.hospital_id,
        )
        checkpoint_path = self.training_config.train_config.checkpoint_dir / "checkpoints" / "latest.pt"
        save_checkpoint(state, checkpoint_path)
