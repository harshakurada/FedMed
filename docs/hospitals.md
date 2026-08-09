# Hospital Nodes & Local Training Architecture (`hospital_nodes/`)

This is a simulated research/portfolio project. **Federated aggregation,
network communication, encryption, and differential privacy are NOT
implemented yet** — this module only prepares the hospital-side pieces
Module 7 (Flower + FedAvg) will later wrap and connect.

## Why three hospital nodes

FedMed's premise (see the root `README.md`) is 3 hospitals collaborating on
a tumor-segmentation model without pooling raw patient data. Before
wrapping anything in Flower, this module proves the actual mechanics work
standalone: each hospital can hold its own data, train its own model, and
produce something a federated client *could* send onward — all with zero
network code, so those mechanics can be debugged without also debugging
gRPC/Flower at the same time.

## How data is partitioned, and why patient-level

`hospital_nodes/partition.py`'s `partition_studies` splits **whole
studies** (patients) round-robin across the 3 hospitals — never individual
slices, and never the same study in two hospitals. Splitting anything finer
than a whole patient would let information about one "patient" leak across
hospital boundaries, which defeats the entire premise of federated
learning over medical data.

Partitioning happens in two steps, in this order:

1. **Module 3/5's existing global train/val split is preserved, not
   recomputed.** `hospital_nodes/simulation.create_hospital_nodes` calls
   `cv_model.brats.split.split_studies` with the same seed/`val_fraction`
   Module 5's centralized baseline used. This is what makes a future
   federated result *comparable* to the centralized baseline — both are
   evaluated against the same held-out studies.
2. Only the **training** share of that split is partitioned across the 3
   hospitals (`hospital_nodes/partition.py`). The validation share is never
   partitioned, never assigned to a hospital, and never used as training
   data anywhere.

Uneven division is handled by round-robin assignment over a seed-shuffled
order (the same technique `cv_model/dataset.py`'s original
`partition_indices` established) — deterministic, and any remainder from
uneven division lands as one extra study on the first few hospitals rather
than being dropped.

Verified with real local data: 9 usable studies → Module 5's split gives 7
train / 2 val → partitioned as Hospital A=3, Hospital B=2, Hospital C=2.

## Privacy boundary

`HospitalNode` (`hospital_nodes/node.py`) keeps its `StudyRecord`s (file
paths — never pixel data) as a private attribute. Its public,
"communication-ready" methods — `get_parameters`, `fit`, `evaluate` —
return only:

- model weights (a `state_dict`, i.e. learned numbers, not patient data)
- scalar/dict training metrics (loss, Dice, IoU, study *counts*)

None of them can return a `StudyRecord`, a file path, or an image tensor —
checked structurally in `hospital_nodes/tests/test_node.py`
(`test_local_training_result_contains_no_raw_dataset_or_tensor_data`).
`HospitalNode.__repr__` reports only a study *count*, never IDs or paths.

## How local training works

`HospitalNode.local_train()` (aliased as `.fit()`) reuses
`cv_model.training.engine.train_one_epoch`/`validate` and
`cv_model.training.trainer.build_optimizer`/`build_scheduler` **unchanged**
— there is no second trainer. It loops `HospitalTrainingConfig.local_epochs`
times (a small number per call — the eventual per-round local-training
budget, a different concept from `TrainingConfig.epochs`, the centralized
baseline's full run length) over that hospital's own `DataLoader`, built
via `cv_model.brats.dataset.build_dataset` on that hospital's own study
partition.

Local validation is **optional** (`HospitalTrainingConfig.local_val_fraction`,
default `0.0`): with the default, a hospital's entire partition trains, and
`evaluate()` returns `None` — evaluation instead happens later against the
shared global validation set from Module 5's split. Setting
`local_val_fraction > 0` carves a further deterministic split *within that
hospital's own partition only*, never touching another hospital's data or
the global holdout.

## Model independence

Every `HospitalNode.__init__` calls `cv_model.model.build_unet_from_params`
itself — three separate model objects, never a shared live reference.
Verified directly: `hospital_nodes/tests/test_node.py` constructs all 3,
loads the *same* initial weights into each (so their **states** can be
equal, matching what a real federated round's initial broadcast would do),
trains only Hospital A, and asserts Hospital B and Hospital C's weights are
bit-identical to before — training one hospital can never affect another.

## Model state interface (what Module 7 will adapt to Flower)

`hospital_nodes/model_state.py` — `get_model_state`, `load_model_state`,
`clone_model_state`, `save_model_state`, `load_model_state_from_disk`,
`states_equal` — all copy-on-read/write, so handing a state to a second
model, or mutating a returned state, can never mutate the source. This is
the interface `HospitalNode.get_parameters()`/`set_parameters()` are built
on, and conceptually maps onto Flower's `NumPyClient`:

| Flower concept | This module's equivalent |
|---|---|
| `get_parameters()` | `HospitalNode.get_parameters()` |
| `set_parameters()` | `HospitalNode.set_parameters(state)` |
| `fit()` | `HospitalNode.fit()` → `LocalTrainingResult` (includes `num_examples`, the FedAvg weighting signal) |
| `evaluate()` | `HospitalNode.evaluate()` → `LocalEvaluationResult` or `None` |

No Flower import exists anywhere in `hospital_nodes/partition.py`,
`node.py`, `model_state.py`, `simulation.py`, or `config.py` — verified in
CI-style by `grep` in this module's final verification. (The pre-existing
`hospital_nodes/client_app.py`, a real Flower `ClientApp` built earlier
during Module 1's node scaffolding, is untouched by this module and still
imports `flwr` directly, as its whole purpose requires — Module 7 should
refactor it to wrap `HospitalNode` rather than its current standalone
synthetic-data training loop, which predates and duplicates what
`HospitalNode.local_train()` now does properly.)

## Hospital checkpoints

Each hospital writes to its own directory, never a shared one:

```
checkpoints/hospitals/
├── hospital_a/checkpoints/latest.pt
├── hospital_b/checkpoints/latest.pt
└── hospital_c/checkpoints/latest.pt
```

(`FEDMED_HOSPITAL_CHECKPOINT_ROOT` overrides the root; rooted under the
same gitignored `checkpoints/` convention Module 4/5 established.) Each
checkpoint (`cv_model.training.checkpoint.CheckpointState`, extended this
module with an optional `hospital_id` field) contains model state,
optimizer state, epoch, training config, and `hospital_id` — never raw MRI
data.

## Running it yourself

```powershell
$env:FEDMED_BRATS_ROOT = "C:\path\to\your\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
.venv\Scripts\python.exe -m pytest hospital_nodes -q      # unit tests, synthetic data only
.venv\Scripts\python.exe -m hospital_nodes.simulation     # small real-data sanity check, all 3 hospitals
```

`hospital_nodes.simulation.run_local_training_sanity_check` deliberately
trains for 1 local epoch on a small patch size — it is a mechanics check,
not a meaningful training run, and does not run automatically as a side
effect of anything else in this project.

## What's still not implemented

Federated aggregation (FedAvg/FedProx), Flower wiring beyond the
pre-existing `client_app.py`, gRPC/TLS network communication, TenSEAL
homomorphic encryption, and differential privacy are all future modules'
work — none of it exists in `hospital_nodes/`'s Module 6 code.
