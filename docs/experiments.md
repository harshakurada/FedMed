# Module 12 Benchmark Experiments

This is a simulated research/portfolio project running on a small local BraTS2020 subset
(9 valid studies) — **these are development-scale numbers proving the pipeline mechanics
work end-to-end, not a claim of clinical accuracy.** A 3D U-Net trained for 1–3 epochs on
7 studies will not, and is not expected to, produce clinically meaningful segmentation —
see "Comparability & limitations" below.

All numbers on this page were produced by actually running the commands shown, on
2026-08-09, against a real local BraTS2020 copy (`FEDMED_BRATS_ROOT` pointed at a
locally-extracted `MICCAI_BraTS2020_TrainingData` directory containing 10 study folders,
9 of which pass discovery/validation — 1 excluded as incomplete, matching Module 3's
existing `on_incomplete_study="exclude"` behavior). No number below is estimated,
interpolated, or carried over unrun from an earlier module — every figure is from this
module's own execution, saved to the `results.json` paths listed per experiment.

## Environment

| | |
|---|---|
| OS | Windows 11 |
| Device | CPU only (no CUDA available) |
| Python | 3.14.7 |
| PyTorch | 2.13.0+cpu |
| MONAI | 1.6.0 |
| TenSEAL | 0.3.17 |
| Flower | 1.33.0 |
| Dataset | Local BraTS2020 subset, 9 valid studies (patient-level split, seed=42): 7 train / 2 val |

## Experiment 1 — Centralized baseline (development-scale)

```powershell
$env:FEDMED_BRATS_ROOT = "<your local BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData>"
$env:FEDMED_TRAIN_EPOCHS = "3"
.venv\Scripts\python.exe -m cv_model.training.run_baseline --experiment-name module12_dev_scale_centralized_baseline
```

`FEDMED_TRAIN_EPOCHS=3` overrides `TrainingConfig`'s production default of 50 — this is a
**development-scale** baseline (dev-scale, matching this module's own instruction not to
run the full 50-epoch official baseline as part of validation), not the eventual official
centralized baseline `docs/training.md` describes. Everything else (architecture,
optimizer, patch size) is the project's default `TrainingConfig`/`BraTSRawConfig`.

Full results: `checkpoints/brats_baseline/metrics/results.json`.

| Metric | Value |
|---|---|
| Train / val studies | 7 / 2 |
| Architecture | `unet_channels=(16,32,64,128,256)`, `unet_strides=(2,2,2,2)`, `unet_num_res_units=2` (production default) |
| Epochs completed | 3 (best at epoch 3, no early stop) |
| Best val Dice | 0.0259 |
| Best val IoU | 0.0134 |
| Best val loss | 0.9679 |
| Per-class Dice | Tumor Core=0.0090, Whole Tumor=0.0625, Enhancing Tumor=0.0063 |
| Checkpoint reproducibility check | Passed — reloaded checkpoint reproduced the recorded Dice exactly |
| Training wall-clock | 104.6s (3 epochs, CPU) |

## Experiments 2–4 — Utility comparison: Plain FedAvg / DP FedAvg / DP+CKKS FedAvg

```powershell
$env:FEDMED_BRATS_ROOT = "<your local BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData>"
$env:FEDMED_DP_ENABLED = "true"
.venv\Scripts\python.exe -m server.federated.dp.run_dp_experiment --experiment-name module12_benchmark_dp_ckks_comparison
```

Single federated round, 1 local epoch per hospital, `server.federated.dp.comparison.
run_utility_comparison` — all 3 arms share the exact same 3 hospitals' fit results and the
same centralized-evaluation function, so only the aggregation/privacy/encryption
treatment differs between rows. DP config: `clip_norm=1.0`, `noise_multiplier=5.0`,
`delta=1e-5` (project defaults). Partition: hospital_a=3, hospital_b=2, hospital_c=2
train studies; 2 shared val studies.

Full DP-arm result: `checkpoints/dp/results/results.json`. Plain-arm numbers below are
from this run's console output (`run_dp_experiment.py` currently only persists the DP arm
to `results.json`; the plain arm is printed but not separately saved — recorded here
verbatim from the actual run rather than left out).

| Arm | Global Dice | Global IoU | Global loss | Epsilon (max, client-level) | Aggregation time |
|---|---|---|---|---|---|
| Plain FedAvg | 0.0182 | 0.0093 | 0.9688 | n/a (no DP) | 0.066s |
| DP FedAvg | 0.0186 | 0.0095 | 0.9800 | 0.9690 | 0.035s |
| DP + CKKS FedAvg | 0.0186 | 0.0095 | 0.9800 | 0.9690 | 35.333s |

**DP FedAvg and DP+CKKS FedAvg report identical Dice/IoU/loss to 4 decimal places** — this
is expected, not a bug: CKKS's own measured numerical error (`docs/homomorphic_encryption.md`,
~6×10⁻⁸ max absolute error) is many orders of magnitude below what would move Dice/IoU/loss
at this precision. CKKS costs real time (encryption + homomorphic aggregation, ~35s here
for 3 hospitals' full model, vs. ~0.04s for the equivalent plaintext sum) but essentially
zero additional utility, exactly as `docs/homomorphic_encryption.md`'s own precision claim
predicts.

**DP costs measurable utility here** even at this tiny scale: DP FedAvg's global loss
(0.9800) is worse than Plain FedAvg's (0.9688) — the expected privacy/utility trade-off,
not a defect (`docs/differential_privacy.md`'s "How to interpret results").

## Comparability & limitations

- **Experiment 1 is not directly comparable to Experiments 2–4 in absolute Dice.**
  Experiment 1 trains 3 epochs on the full pooled 7-study training set; Experiments 2–4
  run a *single federated round* (1 local epoch per hospital, on a 2–3-study partition
  each) — this is `run_utility_comparison`'s own scope (a single-round utility snapshot,
  not a multi-round federated training run), not an apples-to-apples training-budget
  comparison. The 3 FedAvg arms *are* directly comparable to each other (same hospitals,
  same fit results, same evaluation).
- All 4 experiments used the same model architecture (`TrainingConfig`'s production
  default), so the numbers are architecture-comparable even though training-budget differs.
- 9 studies (7 train / 2 val) is a mechanics-verification scale, not a scale that can
  produce clinically meaningful segmentation — see `docs/dataset.md` for why (the full
  ~369-study BraTS2020 release would be needed for a meaningful baseline).
- A full multi-round federated experiment (`python -m server.federated.run_experiment`,
  which supports many rounds) was not re-run as part of this module's benchmarking — it
  is already covered by `docs/federated_training.md` and its own test suite; Module 12's
  scope was the 4 specific experiments above plus the new full-stack integration test
  (`docs/final_validation_report.md`).
