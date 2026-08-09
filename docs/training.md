# Centralized Training Baseline (`cv_model/training/`)

This is a simulated research/portfolio project. **It is not clinically
validated and makes no claim of medical efficacy.**

## What this is, and why it exists

Before implementing federated learning, FedMed needs a **centralized
baseline**: one 3D U-Net, trained normally on all available (pooled) data,
evaluated the same way federated results will later be evaluated. Module 7 implements
FedAvg across the 3 simulated hospital nodes (Module 6); its results are only meaningful
compared against this baseline — otherwise there's nothing to say federated learning
"worked" relative to. See [`docs/federated_training.md`](federated_training.md) for that
comparison.

```
BraTS -> Module 3 preprocessing -> DataLoader -> Module 4 3D U-Net ->
centralized training -> validation -> best checkpoint -> final evaluation ->
results.json + training curves
```

This module does not implement Flower, FedAvg, or anything distributed —
it only adds the *experiment* layer (config validation, a pre-flight
data-leakage check, checkpoint verification, results/plots) on top of
Module 4's existing model/training-loop code, which is reused unchanged.

## Configuration

There is deliberately **no third configuration system**. An experiment is
just `cv_model.training.experiment.ExperimentConfig`: a name plus the two
configs that already existed —

- `cv_model.brats.config.BraTSRawConfig` (Module 3): dataset root, modality/label
  file layout, train/val split seed and fraction, patch size, spacing, batch
  size, num_workers, cache mode
- `cv_model.training.config.TrainingConfig` (Module 4): model architecture
  (`unet_channels`/`unet_strides`/`unet_num_res_units`), optimizer, learning
  rate, weight decay, scheduler, device preference, mixed precision,
  checkpoint directory, validation frequency, early stopping patience

Both are frozen dataclasses, both overridable via `FEDMED_*`/`FEDMED_TRAIN_*`
environment variables (see `docs/dataset.md` and `cv_model/training/config.py`)
— nothing is hard-coded in more than one place.

## How to run it

```powershell
$env:FEDMED_BRATS_ROOT = "C:\path\to\your\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
.venv\Scripts\python.exe -m cv_model.training.run_baseline --experiment-name brats_centralized_v1
```

This validates the configuration, checks for data leakage, trains for
`TrainingConfig.epochs`, verifies the best checkpoint by reloading it into a
fresh model, runs a final evaluation on the validation split, and writes
`results.json` + training-curve plots. **Nothing in this project runs this
automatically** — it only happens when you invoke it.

Before committing to a real run, verify the mechanics on a small scale
first (these do not require or produce an official baseline record):

```powershell
.venv\Scripts\python.exe -m pytest cv_model -q                      # unit tests, no dataset needed
.venv\Scripts\python.exe -m cv_model.training.sanity_check          # one batch: forward/loss/backward/optimizer/metrics
.venv\Scripts\python.exe -m cv_model.training.overfit_check         # DEBUG/SANITY TEST -- can the pipeline learn at all?
```

`overfit_check.py` is explicitly a pipeline diagnostic, not a baseline —
`cv_model.training.experiment.run_baseline_experiment(..., kind=...)` tags
every `results.json` with `"OFFICIAL_CENTRALIZED_BASELINE"` or
`"DEBUG_SANITY_TEST"` so the two are never confused later.

## Data leakage check (runs automatically before training)

`cv_model.training.experiment.check_no_data_leakage` verifies, on the
actual discovered dataset, before any training happens:

1. The train and validation study-ID sets are disjoint (patient-level split
   from Module 3 — re-verified here, not re-computed with a new seed).
2. The validation transform pipeline contains no `Rand*` step (checked by
   inspecting the actual `Compose` object's transform list, not assumed).

If either check fails, it raises `DataLeakageError` and training does not
start. (What it does *not*, and structurally cannot, verify at runtime: that
no validation-set information ever influences an optimizer step — that's
guaranteed by `engine.validate()` always running under `torch.no_grad()`
and never being passed to `optimizer.step()`, which is a property of the
code, not something checked at run time.)

## Metrics: what's reported and how

- **Dice** and **IoU**, both per-region (Tumor Core, Whole Tumor, Enhancing
  Tumor — see `cv_model/brats/labels.py`) via MONAI's `DiceMetric`/`MeanIoU`,
  `include_background=True`, `reduction="mean_batch"`. These are multi-label
  *overlapping* regions, not mutually-exclusive classes, so there's no
  separate background channel to include or exclude the way there would be
  for single-label multi-class segmentation.
- The scalar "Dice"/"IoU" reported in logs and `history.json` is the
  **macro mean across the 3 regions** (each region weighted equally,
  regardless of its voxel count). Per-region breakdown is reported
  separately in `results.json`'s `per_class_dice`/`per_class_iou`.
- **Empty-class handling**: `ignore_empty=True` (MONAI default) means a
  region absent from both prediction and ground truth in a given *sample*
  doesn't count against that sample. If a region is absent across an
  *entire batch/validation set*, MONAI has nothing left to average and
  reports **0.0** for it — not NaN, and not silently omitted. Read a 0.0
  per-region score in context: it can mean "the model missed this region
  everywhere" or "this region never appeared in the evaluated set" —
  `num_val_studies` in `results.json` tells you how much data that score
  is actually based on. Verified in `cv_model/training/tests/test_metrics_semantics.py`.
- **Best checkpoint selection**: validation Dice (macro mean). Whichever
  epoch has the highest validation Dice becomes `checkpoints/best.pt`.

## Checkpoint validation and final evaluation

After training, `cv_model.training.final_evaluation.run_final_evaluation`
loads `checkpoints/best.pt` into a **freshly constructed** model (not the
in-memory one from training) and re-runs validation. `load_state_dict` is
strict, so an architecture mismatch fails loudly here rather than silently
producing garbage predictions. The re-evaluated Dice is compared against
what training itself recorded (tolerance `1e-3` — MONAI/PyTorch ops aren't
bit-identical run to run even with the same seed); `results.json`'s
`checkpoint_reproduced_recorded_dice` records whether they matched.

## Where results are saved

Rooted at `TrainingConfig.checkpoint_dir` (default `./checkpoints/brats_baseline`,
override via `FEDMED_TRAIN_CHECKPOINT_DIR`) — the existing Module 4
convention, not a second parallel output tree:

```
<checkpoint_dir>/
├── checkpoints/
│   ├── latest.pt     # every epoch
│   └── best.pt        # highest validation Dice so far
├── history/
│   └── history.json   # per-epoch: loss, Dice, IoU, LR, duration
├── metrics/
│   └── results.json   # the one-shot experiment record (see below)
├── plots/
│   ├── loss_curve.png
│   ├── dice_curve.png
│   └── iou_curve.png
└── inference/          # cv_model.training.inference.run_inference() output
    └── <study_id>.png  # MRI + ground truth + prediction, per inspected study
```

All of it is gitignored (`checkpoints/` in `.gitignore`) — never commit a
checkpoint or a results file unless explicitly asked to later.

`results.json` contains: experiment name, timestamp, seed, dataset
root/split counts/split seed, model config, training config, preprocessing
config, best epoch, best/final validation Dice+IoU, per-region breakdown,
Dice semantics documentation (embedded, not just in this file),
checkpoint-reproducibility flag, training duration, device, and
python/torch/monai versions. It never contains raw MRI pixel data or
patient-identifying information — see `cv_model/training/tests/test_results.py`'s
structural check.

## How to interpret the training curves

`cv_model/training/plots.py` (reuses matplotlib, the same library
`cv_model.brats.inspect_slices` already uses — no new plotting dependency)
reads `history.json` and produces 3 PNGs:

- **loss_curve.png** — train vs. validation loss. Diverging lines (train
  falling, validation rising) is the classic overfitting signature.
- **dice_curve.png** / **iou_curve.png** — validation Dice/IoU per epoch
  (only epochs where validation actually ran, per `val_frequency`).

## Qualitative inspection

`cv_model.training.inference.run_inference(...)` (Module 4, unchanged)
loads the best checkpoint and saves an MRI + ground-truth + prediction
comparison PNG for a handful of validation studies — deliberately not run
over the whole validation set. Use this to confirm predictions are
spatially plausible, not just that the scalar metrics look reasonable.

## Preparing for the federated comparison (Module 6)

For the eventual centralized-vs-FedAvg comparison to mean anything, both
must share: the same model architecture (`unet_channels`/`strides`/
`num_res_units`), the same preprocessing (`cv_model.brats.transforms`),
the same label representation (`cv_model.brats.labels`), and the same
validation protocol (Dice/IoU, `include_background=True`,
`reduction="mean_batch"`). `results.json`'s `model_config`/
`preprocessing_config`/`dice_semantics` fields exist specifically so a
future federated results file can be diffed against them to confirm this
holds. **FedAvg itself is not implemented in this module.**
