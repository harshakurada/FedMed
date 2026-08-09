# Federated Learning + FedAvg (`server/federated/`, `hospital_nodes/client_app.py`)

This is a simulated research/portfolio project. **It is not clinically validated and
makes no claim of medical efficacy or of being privacy-preserving against an untrusted
server yet** (see "Security boundary" below).

## What federated learning means in FedMed

Instead of pooling the 3 hospitals' MRI studies into one dataset (Module 5's centralized
baseline), each hospital trains its own copy of the shared 3D U-Net on only its own
patient-level partition (Module 6), and only **model weights and scalar metrics** --
never images, labels, or file paths -- cross to a central aggregator:

```
Global model
     |
   broadcast
  /    |    \
Hosp A Hosp B Hosp C     <- each trains locally, on its own partition only
  \    |    /
   FedAvg (weighted by each hospital's num_examples)
     |
Updated global model  ->  next round
```

## What FedAvg does here

Flower's own `FedAvg` strategy (`flwr.server.strategy.FedAvg`, unmodified -- see
`server/federated/strategy.py`) averages the 3 hospitals' returned weights, **weighted by
each hospital's own training-set size** (`flwr.server.strategy.aggregate.aggregate`), not
a plain 3-way average. A hospital with more studies pulls the global model further toward
its own update. Custom scalar metrics (e.g. each hospital's own `train_dice`) are
aggregated the same way via `server/federated/evaluation.py`'s `weighted_average`.

## How the three hospitals participate

Each hospital is a `HospitalNode` (Module 6, `hospital_nodes/node.py`) wrapped by
`hospital_nodes/client_app.py`'s `HospitalNodeClient(NumPyClient)` -- a thin adapter that
only converts between Flower's NDArrays wire format and the node's own
`get_parameters`/`set_parameters`/`fit`/`evaluate` interface. No local-training or
data-partitioning logic lives in the Flower-facing file; it's all reused unchanged from
Module 6.

## Local training

`HospitalNode.fit()` (Module 6, reused unchanged) trains for `local-epochs` (configurable,
default 1) on that hospital's own patient-level partition only, reusing
`cv_model.training`'s optimizer/scheduler/train-loop unchanged. `local-epochs` is a
per-round local-training budget, a different concept from a full centralized run's total
epoch count.

## Evaluation strategy: centralized (default) vs. distributed (optional)

Two independent, clearly-labeled evaluation paths exist -- never conflated:

- **Centralized ("global") evaluation** -- the default, and the one compared against the
  centralized baseline. After each round's aggregation, the server evaluates the
  aggregated global model against Module 5's **held-out global validation split** (never
  a hospital's training data, never recomputed). Reported as `global_loss`/`global_dice`/
  `global_iou` in `history.json`. Implemented in `server/federated/evaluation.py`'s
  `build_centralized_evaluate_fn`, which reuses `cv_model.training.engine.validate`
  unchanged -- the same function the centralized baseline and `HospitalNode.evaluate()`
  already use.
- **Distributed ("client") evaluation** -- optional, off by default
  (`fraction-evaluate=0.0`). Enabling it requires `local-val-fraction > 0` too (validated
  by `server/federated/config.py`'s `validate_federated_config`): each hospital's
  `evaluate()` then runs against its own local validation carve-out (a further split
  *within* that hospital's own training partition -- never touching another hospital's
  data or the global holdout), sample-weighted-averaged into `client_dice`/`client_iou`.
  These are **never** called "global" -- they're a different, smaller, non-held-out set.

## What crosses the Flower interface, and what never does

**Sent to the server:** model weights (a state_dict converted to NDArrays), and scalar
metrics (`train_loss`/`train_dice`/`train_iou`/`hospital_id`, plus `val_dice`/`val_iou`
only when distributed evaluation is enabled).

**Never sent:** raw MRI volumes, segmentation labels, file paths, or `DataLoader`s. The
server never imports `cv_model.brats.discovery`/`dataset` for a hospital's *training*
partition -- only for its own centralized-evaluation studies (Module 5's held-out set,
architecturally the auditor's own benchmark, not a hospital's private data). Verified
structurally by `server/tests/test_client_proxy.py` and
`hospital_nodes/tests/test_client_app.py` (results never carry a `StudyRecord` or
`torch.Tensor`), mirroring Module 6's existing no-leakage tests one layer up.

## Why round orchestration is in-process, not a live network run

Flower 1.33 (the installed version) offers a live deployment (`flower-superlink` +
`flower-supernode` per hospital + `flwr run`) -- real gRPC over sockets -- and a
Simulation Engine (`flwr.simulation.run_simulation`) whose only backend requires the `ray`
package. **gRPC/TLS are explicitly out of scope for this module** (a later module's job),
and `ray` is not one of this module's approved technologies (Python/PyTorch/MONAI/Flower
only). So `server/federated/experiment.py` calls Flower's own strategy/client objects
directly, in-process: `FedAvg`'s `configure_fit`/`aggregate_fit`/`evaluate`/
`configure_evaluate`/`aggregate_evaluate` never touch the network themselves -- they only
read/write `FitRes`/`EvaluateRes` objects. The only networked piece in a live deployment
is `ClientProxy.fit()/.evaluate()` dispatching over gRPC to a remote process;
`server/federated/client_proxy.py`'s `InProcessClientProxy` replaces only that one call
with a direct Python call to a local `Client` (obtained the standard way, via
`NumPyClient.to_client()`). Everything else that runs -- `FedAvg` construction, client
sampling via `SimpleClientManager`, weighted aggregation -- is genuine, unmodified Flower
code.

`hospital_nodes/client_app.py`'s `ClientApp` and `server/server_app.py`'s `ServerApp` are
still built correctly (same strategy factory, same real hospital-node wiring) so a later
module can actually deploy them live once gRPC/TLS is in scope -- but the round loop this
module builds, tests, and documents runs in-process.

## How to run it

**Smoke test** (tiny synthetic data, no BraTS download, ~2 seconds):

```powershell
.venv\Scripts\python.exe -m pytest server hospital_nodes -q
```

**Full federated experiment** (real local BraTS2020 data):

```powershell
$env:FEDMED_BRATS_ROOT = "C:\path\to\your\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
.venv\Scripts\python.exe -m server.federated.run_experiment --experiment-name federated_v1
```

Configurable via `FEDMED_FED_*` environment variables (`server/federated/config.py`) --
e.g. `$env:FEDMED_FED_NUM_ROUNDS = "5"`, `$env:FEDMED_HOSPITAL_LOCAL_EPOCHS = "2"`.
**Nothing in this project runs this automatically** -- it only happens when invoked.

A live multi-process deployment (`flower-superlink` + 3x `flower-supernode` +
`flwr run`) is intentionally not documented here -- it requires gRPC, out of scope this
module. `server/server_app.py` and `hospital_nodes/client_app.py` are ready for a later
module to wire that up once TLS is in scope.

## Where checkpoints/results/plots are stored

```
checkpoints/federated/
    checkpoints/
        initial_global.pt   # before round 1
        latest_global.pt    # overwritten every round
        best_global.pt      # overwritten when global Dice improves
    history/history.json    # one record per round: per-hospital + aggregated + global metrics
    metrics/results.json    # one-shot summary, comparable to checkpoints/brats_baseline/metrics/results.json
    plots/                  # global loss/Dice/IoU vs. round (matplotlib PNGs)
```

Global checkpoints have no single optimizer (each hospital keeps its own local optimizer
state, saved separately under `checkpoints/hospitals/hospital_x/`) -- `optimizer_state_dict`
is an empty dict there, and `hospital_id=None` marks a checkpoint as global rather than
per-hospital. Never committed to Git (`checkpoints/` is gitignored); never overwrites the
centralized baseline's `checkpoints/brats_baseline/`.

## Baseline comparison

If `checkpoints/brats_baseline/metrics/results.json` exists (Module 5's centralized
baseline), `run_federated_experiment` automatically compares against it via
`server/federated/results.py`'s `compare_to_baseline` -- which first asserts the two runs
used the same model architecture (raising rather than silently comparing incompatible
experiments), then reports the Dice/IoU deltas.

## Security boundary

**Current (Module 7):** federated aggregation is implemented with real FedAvg; raw data
stays architecturally local to each hospital (never imported, never crosses the Flower
interface); model updates are exchanged as plain weights and scalar metrics.

**Not yet implemented:** encryption, TLS, secure aggregation, differential privacy. A
server operator (or anyone intercepting the in-process/future-network channel) can
currently inspect the plaintext weight updates each hospital sends. This system is **not**
currently privacy-preserving against an untrusted server or network observer -- that's
later modules' work.
