# BraTS Dataset Pipeline (`cv_model/brats/`)

## The dataset is not in this repository

BraTS is **not** committed to Git and is **never auto-downloaded** by this
project. You must obtain it yourself (e.g. from Kaggle or the official
BraTS/Synapse release) and point the pipeline at your local copy via the
`FEDMED_BRATS_ROOT` environment variable. The dataset stays on your machine,
outside source control, under the gitignored `data/` directory (or anywhere
else you choose).

This is a simulated research/portfolio project. It does **not** claim
HIPAA/GDPR compliance, clinical validation, or medical approval of any kind.

## Expected directory structure

This pipeline was built and verified against a **BraTS2020** Kaggle mirror.
Other BraTS releases (2019/2021/...) may name files differently -- if
`cv_model.brats.discovery` reports every study as incomplete, check your
actual filenames against `BraTSRawConfig.modalities`/`label_suffix` below
before assuming the data is bad.

```
<FEDMED_BRATS_ROOT>/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_flair.nii
│   ├── BraTS20_Training_001_t1.nii
│   ├── BraTS20_Training_001_t1ce.nii
│   ├── BraTS20_Training_001_t2.nii
│   └── BraTS20_Training_001_seg.nii
├── BraTS20_Training_002/
│   └── ...
├── name_mapping.csv        # not used by this pipeline
└── survival_info.csv       # not used by this pipeline
```

Each patient/study is one subdirectory; each file is `<study_id>_<name><ext>`.
Both `.nii` and `.nii.gz` are accepted (`BraTSRawConfig.file_extensions`).

There is intentionally no separate "validation" directory: BraTS's own
official validation split ships without ground-truth labels (it exists for
the challenge leaderboard), so it's useless for our supervised train/val
split. Our validation set is instead held out from the labeled training
studies -- see [Train/validation split](#trainvalidation-split) below.

## Configuration (`cv_model/brats/config.py`)

`BraTSRawConfig` is a frozen dataclass; every field is overridable via a
`FEDMED_BRATS_*` environment variable (no path is hard-coded anywhere else
in the pipeline):

| Field | Env var | Default | Meaning |
|---|---|---|---|
| `root` | `FEDMED_BRATS_ROOT` | `./data/brats2020/raw` | Directory containing one subdirectory per patient |
| `val_fraction` | `FEDMED_BRATS_VAL_FRACTION` | `0.2` | Fraction of studies held out for validation |
| `seed` | `FEDMED_BRATS_SEED` | `42` | Split determinism |
| `batch_size` | `FEDMED_BRATS_BATCH_SIZE` | `2` | Training DataLoader batch size |
| `num_workers` | `FEDMED_BRATS_NUM_WORKERS` | `0` | DataLoader worker processes (see [Windows note](#windows-multiprocessing-note)) |
| `cache_mode` | `FEDMED_BRATS_CACHE_MODE` | `none` | `none` (`monai.data.Dataset`) or `cache` (`monai.data.CacheDataset`) |
| `cache_rate` | `FEDMED_BRATS_CACHE_RATE` | `1.0` | Fraction of studies to cache in RAM, if `cache_mode=cache` |
| `validation_sample_size` | `FEDMED_BRATS_VALIDATION_SAMPLE_SIZE` | `3` | How many studies `validate_studies()` loads pixel data for |

`modalities`, `label_suffix`, `file_extensions`, `pixdim`, `patch_size` are
set in code (not env vars) since changing them changes the model's expected
input shape -- they're meant to be deliberate, reviewed changes.

## Dataset discovery and validation

**Discovery** (`cv_model/brats/discovery.py`) walks the filesystem only --
it never loads pixel data. For each study directory it checks that every
configured modality file and the label file exist. `BraTSRawConfig.root`'s
real data (10 patients) includes one genuinely incomplete study
(`BraTS20_Training_010`, missing `t1ce`/`t2`) -- discovery either raises
(`on_incomplete_study="raise"`, the default) listing every incomplete study
found, or excludes them while still reporting which ones and why
(`on_incomplete_study="exclude"`). It never silently drops a study without
telling you.

**Validation** (`cv_model/brats/validation.py`) goes one level deeper on a
small *sample* of studies (`validation_sample_size`, default 3) -- not the
whole dataset -- actually loading pixel data to check for NaN/Inf values,
label values outside the documented set, and modality/label volumes whose
shapes don't match.

Run validation over your dataset:

```bash
.venv\Scripts\python.exe -c "
from cv_model.brats.config import DEFAULT_CONFIG
from cv_model.brats.discovery import discover_studies
from cv_model.brats.validation import validate_studies
from dataclasses import replace

config = replace(DEFAULT_CONFIG, on_incomplete_study='exclude')
result = discover_studies(config)
print(f'{len(result.valid)} valid, {len(result.incomplete)} incomplete')
report = validate_studies(list(result.valid), config, sample_size=len(result.valid))
print(report.summary())
"
```

## Label handling

Documented in full in `cv_model/brats/labels.py`. Verified directly against
this project's real data (`BraTS20_Training_001_seg.nii`): voxel values
present are exactly `{0, 1, 2, 4}`:

- `0` = background
- `1` = NCR/NET (necrotic and non-enhancing tumor core)
- `2` = ED (peritumoral edema)
- `4` = ET (enhancing tumor) -- **note: `3` is never used**, a long-standing
  artifact of the BraTS challenge's labeling history

`ConvertBraTSLabelsd` remaps these into the 3 official overlapping BraTS
evaluation regions (multi-label, not mutually-exclusive classes):

- Tumor Core (TC) = `{1, 4}`
- Whole Tumor (WT) = `{1, 2, 4}`
- Enhancing Tumor (ET) = `{4}`

This is a **different** convention from `cv_model/transforms.py`'s
Decathlon-oriented converter (which relabels ET as `3` and groups Tumor
Core as `{2, 3}`) -- the two are kept as separate, explicit converters
rather than one that guesses which convention applies.

## Train/validation split

`cv_model/brats/split.py` splits whole **studies** (patients), never
individual slices -- the same patient's data can never appear in both
train and validation, which would leak information and inflate the
reported validation Dice score. The split is deterministic for a given
`seed`, and reports total/train/val counts, the ratio, and the seed used
(see `SplitResult.summary()`).

## Preprocessing pipeline (`cv_model/brats/transforms.py`)

| Stage | Purpose |
|---|---|
| `LoadImaged` + `EnsureChannelFirstd` | Load each modality + label NIfTI file |
| `ConcatItemsd` | Stack the 4 modalities into one `(4, H, W, D)` "image" tensor |
| `ConvertBraTSLabelsd` | Raw labels `{0,1,2,4}` -> 3-channel TC/WT/ET |
| `Orientationd` (RAS) | Consistent anatomical orientation across scanners |
| `Spacingd` (1.5mm isotropic) | Consistent physical voxel size across patients |
| `CropForegroundd` | Drop the (mostly zero) padding around the head |
| `SpatialPadd` | Pad back up to `patch_size` if foreground-cropping left a volume smaller than the patch in any axis (real, observed on this dataset) |
| `RandCropByPosNegLabeld` (train only) | Random tumor-biased 128x128x64 patches (2 per volume) |
| `RandFlipd` x3, `RandShiftIntensityd` (train only) | Realistic MRI augmentation -- axis flips + small intensity jitter; no rotation/elastic warps that would distort tumor geometry |
| `ResizeWithPadOrCropd` (val only) | One deterministic fixed-size crop/pad per volume |
| `NormalizeIntensityd` | Per-channel z-score normalization over nonzero voxels |
| `EnsureTyped` | Convert to `torch.Tensor` |

Validation uses a separate, non-random pipeline (`get_val_transforms`) so
validation metrics are reproducible run to run.

## How to run the sanity check

Loads a **small, capped sample** of studies (3) through the full pipeline
-- discovery, validation, transforms, and one DataLoader batch. No training,
and it does not process the whole dataset:

```powershell
$env:FEDMED_BRATS_ROOT = "C:\path\to\your\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
.venv\Scripts\python.exe -m cv_model.brats.sanity_check
```

## How to inspect a study visually

Saves a PNG (4 modality slices + 3 label-region masks, one middle axial
slice) -- a dev-time utility, not the React dashboard:

```powershell
$env:FEDMED_BRATS_ROOT = "C:\path\to\your\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
.venv\Scripts\python.exe -m cv_model.brats.inspect_slices --study-id BraTS20_Training_001 --out slices.png
```

## Running the tests

All tests use small synthetic (non-medical) fixtures generated on the fly
(`cv_model/brats/tests/conftest.py`) -- none require the real BraTS dataset:

```powershell
.venv\Scripts\python.exe -m pytest cv_model/brats/tests -v
```

## Windows multiprocessing note

`num_workers` defaults to `0` (single-process loading). PyTorch/MONAI
DataLoader workers use spawn-based multiprocessing on Windows, which
re-imports the launching script in each worker process -- raising
`num_workers` above 0 is only safe from a script with an
`if __name__ == "__main__":` guard around the training loop.

## Common errors

- **`FileNotFoundError: BraTS dataset root does not exist`** -- `FEDMED_BRATS_ROOT` isn't set or points at the wrong folder. This project never auto-downloads BraTS.
- **`IncompleteStudyError`** -- one or more study directories are missing a modality or label file. The error lists every affected study; pass `on_incomplete_study="exclude"` to proceed without them.
- **`ValueError: The size of the proposed random crop ROI is larger than the image size`** -- would happen without `SpatialPadd`; already handled in `get_train_transforms`. If you lower `patch_size` significantly this shouldn't recur, but if you see it again after changing `pixdim`, check the resulting foreground-cropped shape.
- **`ValueError: Duplicate study ID`** -- two directories (possibly across merged BraTS releases) share the same study ID.

## Hardware considerations

- Each raw study is ~77MB (4 modalities + label, `240x240x155` float volumes). `cache_mode="cache"` at `cache_rate=1.0` on the full ~369-study BraTS2020 release would need tens of GB of RAM -- start with `cache_mode="none"` (re-reads from disk each epoch) or a low `cache_rate` until you've checked your machine's available RAM.
- `batch_size` (default 2) and `patch_size` (default `128x128x64`) are the two knobs to shrink first if you hit GPU/CPU memory limits -- both are configurable, nothing is hard-coded.
- This project's local dataset copy has only 10 studies (9 usable) -- enough to validate the pipeline end to end, not enough for meaningful training. Module 4 will need the fuller BraTS2020 release (~369 studies) for that.
