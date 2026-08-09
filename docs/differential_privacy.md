# Differential Privacy for Federated Model Updates (Module 10)

This is a simulated research/portfolio project. **It is not clinically validated, and no
formal privacy guarantee is claimed beyond exactly what is implemented and accounted for
below.**

## Privacy unit — read this first

**This module implements client-level (hospital-level) Differential Privacy. It does
NOT provide patient-level, example-level, or record-level guarantees.**

Clipping bounds one hospital's *total* per-round contribution to the global model
(‖Δ‖₂ ≤ C, where Δ is that hospital's update — see "What is clipped" below). It does
**not** bound any individual patient's, or any individual training patch's, marginal
contribution *within* that hospital's local training. `cv_model/brats/` partitions data
by patient/study (Module 3), but each patient's MRI volume is cropped into multiple
training patches per epoch (`cv_model.brats.transforms`) — many samples per patient, and
a hospital's Δ already reflects however many patients and patches it trained on. A
hospital with 50 patients and a hospital with 2 patients face the *same* clipping bound —
this protects "did hospital X participate, and roughly how much did its data move the
model," not "was patient Y's specific data used." If patient-level accounting were needed,
it would require per-patient gradient isolation during local training (effectively
per-example DP-SGD, which the approved technology stack for this module explicitly
excludes — Opacus/TF-Privacy provide that and are not permitted here). This is not
attempted, and no such claim is made anywhere in this project.

## Why Differential Privacy is added

Modules 8–9 protect the *transport* (TLS/mTLS) and the *server-side computation*
(CKKS — the aggregation server never decrypts an individual update). Neither protects
what the **final decrypted aggregate itself** could leak about a hospital's
participation across many rounds (e.g., via later model-inversion-style analysis of how
the global model changed). DP adds a third, independent, complementary protection at the
source: each hospital bounds and obscures its own per-round contribution *before*
anything leaves the hospital.

## What is clipped, where, and why

**What:** `Δ = post_training_params − pre_round_global_params` — the hospital's actual
contribution this round. Confirmed by reading the code (not assumed): both
`hospital_nodes/client_app.py` (Module 7) and `server/federated/encrypted/
run_encrypted_round.py` (Module 9) produce the **full post-training model parameters**
as "the update," not a gradient and not an explicit delta — `flwr.server.strategy.
aggregate.aggregate` and Module 9's homomorphic aggregation both average these raw
values directly. Clipping the raw parameter values themselves would be meaningless (a
trained model's weights aren't "small"; clipping them would just damage the model). So
Module 10 computes Δ itself, clips *that*, and reconstructs `dp_params = pre_round_params
+ noised_clipped_delta` — a drop-in replacement for the raw parameters everywhere
downstream (same shape, same interface).

**Where/when:** at the hospital (`server/federated/dp/dp_update.py::apply_dp_mechanism`),
immediately after `HospitalNode.fit()` completes and before `encrypt_model_update`
(Module 9) — see `server/federated/encrypted/run_encrypted_round.py`.

**Why this representation:** it's the standard client-level-DP construction from the
federated-learning literature (McMahan et al., *Learning Differentially Private
Recurrent Language Models*, 2017) — clip the client's *contribution*, not its absolute
state. One global L2 clip over the entire flattened parameter vector (not per-layer),
matching that construction and Abadi et al.'s DP-SGD convention.

```
delta_clipped = delta * min(1, C / ||delta||_2)
```

Numerically safe: a zero-norm delta is returned unchanged (no division by zero); NaN/Inf
anywhere in delta raises immediately (`server/federated/dp/clipping.py`) — never
silently propagated into a corrupted model update.

## How noise is calibrated

`noise_std = noise_multiplier * clip_norm`; `Δ_dp = Δ_clipped + N(0, noise_std² · I)`,
added **locally by each hospital, before encryption** (`server/federated/dp/noise.py`).
This ordering is required both by the task and by Module 9's own design: the server never
decrypts anything, so there is no plaintext-aggregate stage where a *central* noise-adder
could even exist. Calibrating noise std as a multiple of `clip_norm` (rather than an
absolute value) is what makes the epsilon formula below depend only on
`noise_multiplier` and `delta` — the clip norm is the sensitivity, already accounted for.

`rng` is always an explicit parameter, never a hard-coded seed: tests pass a seeded
`np.random.default_rng(seed)`; `run_encrypted_round_smoke_test`'s production default
(when `dp_rng` is omitted) is an **unseeded** `np.random.default_rng()`. The two code
paths never overlap — a fixed seed can never accidentally end up in a real experiment.

## What epsilon and delta mean, for this mechanism specifically

`delta` is the configured target failure probability (must be in `(0, 1)`, conventionally
much smaller than 1). `epsilon` is computed, not chosen — `server/federated/dp/
accountant.py::compute_epsilon` uses the **classical Gaussian mechanism bound** (Dwork &
Roth, *The Algorithmic Foundations of Differential Privacy*, Appendix A, Theorem A.1):

```
epsilon = sqrt(2 * ln(1.25 / delta)) / noise_multiplier
```

This bound is proven for `epsilon < 1`; `compute_epsilon` returns `valid_range=False`
(not a hidden/incorrect number) when a configuration falls outside that range — e.g. the
naive `noise_multiplier=1.0` gives `epsilon≈4.84` (`valid_range=False`); the project's
default `noise_multiplier=5.0` gives `epsilon≈0.969` (`valid_range=True`), chosen for
exactly this reason, not arbitrarily.

## How privacy accumulates across rounds

**Basic sequential composition** (Dwork & Roth, Theorem 3.16): T rounds of the same
hospital's participation compose to `(T·epsilon₀, T·delta₀)`. This is real and correct,
but **intentionally conservative, not tight** — a Rényi-DP/moments-accountant approach
would give tighter `O(√T)` scaling instead of this method's `O(T)`. Implementing a
correct moments accountant from scratch (without Opacus/TF-Privacy, both explicitly out
of scope for this module) is exactly the kind of "fake precision" this project's own
instructions warn against — basic composition is simple enough to verify by hand and
never overstates the achieved privacy. `PrivacyAccountant` tracks this **per hospital**,
not globally (composition is about what repeated observation of *one* hospital's updates
reveals — hospital A's and hospital B's budgets are independent) and **never resets**
between rounds. A hospital that fails/drops a round (Module 8) is never charged for it —
the accountant is only ever called for a hospital that actually produced a DP update.

Optional enforcement: `DPConfig.max_epsilon` + `budget_enforcement_enabled=True` makes
`PrivacyAccountant.record_round` raise `PrivacyBudgetExceededError` rather than silently
continue once a hospital's cumulative epsilon would exceed the configured limit.

## How DP interacts with CKKS (Module 9)

```
local update -> DP protection (clip + noise) -> CKKS encryption -> gRPC/TLS -> encrypted aggregation
```

Never the reverse (encrypt, then try to noise ciphertext) — CKKS supports plaintext-
scalar homomorphic operations but adding *calibrated Gaussian noise* homomorphically
would need per-slot random ciphertexts with no clean sensitivity story, and isn't
attempted. DP happens once, in plaintext, entirely inside the hospital's own process,
*before* `encrypt_model_update` is ever called — the raw (non-DP) update and the
DP-protected update's *intermediate* plaintext values never leave the hospital process;
only ciphertext bytes and approved metadata cross the gRPC boundary (Module 8's
`SubmitEncryptedUpdate`, unchanged).

`server/federated/encrypted/run_encrypted_round.py::run_encrypted_round_smoke_test`
gained an **optional** `dp_config` parameter; `dp_config=None` (the default) is
byte-for-byte Module 9's original behavior — confirmed by Module 9's own test suite
still passing unmodified. When DP is enabled, the function's internal plaintext-vs-CKKS
comparison was updated to compare against the *DP-protected* plaintext aggregate (not
the raw one) — otherwise `max_abs_error` would measure DP's noise magnitude (large, and
*expected*) instead of what that comparison exists to isolate: CKKS's own numerical
error, which stays at the same ~1e-7 scale as Module 9's whether or not DP is enabled.

## DP vs. TLS (Module 8) — neither replaces the other

TLS protects data in transit; it says nothing about what the (already-decrypted, at the
transport layer) content reveals. DP protects what the *content itself* reveals about a
hospital's participation, regardless of how securely it was transmitted. A system with
TLS but no DP still fully exposes each hospital's raw contribution to statistical
analysis by anyone who later sees the aggregate; a system with DP but no TLS would leak
the (already-noised) update to network eavesdroppers. This project uses all three layers
(TLS, CKKS, DP) because each addresses a different point of exposure.

## Which FedAvg paths get DP

Module 7's plaintext path (`server/federated/experiment.py`, `hospital_nodes/
client_app.py`) is **completely untouched** by this module — modifying it for a
DP-without-encryption combination risked entangling the existing, tested plaintext
FedAvg for a combination outside this module's core ask (the task's own explicit
fallback permits this: "at minimum preserve Plain FedAvg and DP + encrypted FedAvg").
The 3-way utility comparison below still produces a "DP FedAvg, no encryption" arm — by
applying the same `apply_dp_mechanism` and weighted-averaging in plaintext via
`flwr.server.strategy.aggregate.aggregate` (reused) — without touching
`hospital_nodes/client_app.py` at all.

## Utility comparison — real measured numbers

`server/federated/dp/comparison.py::run_utility_comparison` builds 3 real hospitals once
and produces all three arms from the *same* fit results and the *same* centralized
evaluation function (`server/federated/evaluation.py::build_centralized_evaluate_fn`,
Module 7, reused unchanged):

| Arm | Privacy mechanism | Security mechanism |
|---|---|---|
| Plain FedAvg | none | none |
| DP FedAvg | client-level DP (clip+noise) | none |
| DP + CKKS FedAvg | client-level DP (clip+noise) | CKKS homomorphic aggregation |

A real run on tiny synthetic data (`clip_norm=0.5`, `noise_multiplier=5.0`,
`delta=1e-5`, one round) measured: **Plain FedAvg** global Dice ≈ 0.233, loss ≈ 0.657;
**DP FedAvg** global Dice ≈ 0.204, loss ≈ 0.784, epsilon ≈ 0.969; **DP+CKKS FedAvg**
Dice/loss matched DP FedAvg to ~1e-7 (CKKS's own precision — negligible extra
degradation beyond DP's noise). These are illustrative dev-scale numbers (tiny synthetic
data, 1 local epoch), not a claim about real BraTS-scale utility — see "How to run
experiments" to reproduce or extend them.

## Metric labeling — never conflated

- **Privacy metrics**: `epsilon`, `delta`, `clip_norm`, `noise_multiplier`, privacy unit,
  budget status.
- **Utility metrics**: Dice, IoU, training/validation loss.
- **Security mechanisms** (not privacy metrics): CKKS (computation-time
  confidentiality), TLS (transport confidentiality).

`Dice is never a privacy metric. Epsilon is never a utility metric. CKKS/TLS are
mechanisms, not accounting results.` `server/federated/dp/results.py`'s
`DPRoundResult` keeps these in explicitly-separated, explicitly-named fields.

## Limitations

- Honest-but-curious threat model only (matches Module 9's) — no protection against a
  malicious hospital submitting a crafted update, or a malicious server deviating from
  the protocol.
- Basic composition is conservative, not tight — real cumulative epsilon after many
  rounds is almost certainly better (lower) than what this accountant reports, but this
  accountant will never *understate* the privacy cost.
- Client-level, not patient-level (see "Privacy unit" above) — a hospital's internal
  patient population is not individually protected by this mechanism.
- No protection against what the *final, legitimately decrypted* global model reveals
  through downstream use (e.g. deploying it) — DP here only bounds what one round's
  *contribution* could reveal, not lifetime model-usage risk.

## How to run

```powershell
.venv\Scripts\activate
.venv\Scripts\python.exe -c "import tenseal; print(tenseal.__version__)"   # Module 9 dependency, unchanged

# Module 10 unit tests
.venv\Scripts\python.exe -m pytest server\tests -k dp -v

# Privacy accounting tests specifically
.venv\Scripts\python.exe -m pytest server\tests\test_dp_accountant.py -v

# DP + CKKS smoke test
.venv\Scripts\python.exe -m pytest server\tests\test_dp_ckks_integration.py server\tests\test_dp_three_hospital_smoke.py -v

# Three-hospital DP experiment / baseline comparison (real local BraTS2020 data, single round)
$env:FEDMED_BRATS_ROOT = "C:\path\to\your\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
.venv\Scripts\python.exe -m server.federated.dp.run_dp_experiment

# Baseline comparison (all of Modules 1-10)
.venv\Scripts\python.exe -m pytest -q
```

## How to interpret results

`DPRoundResult`/`DPExperimentResults` (`server/federated/dp/results.py`) record, per
round: the privacy parameters actually used (never a fabricated epsilon — always derived
from the configured mechanism via `compute_epsilon`), `cumulative_epsilon` (never reset),
`budget_status` (`"ok"`/`"exceeded"`), and utility metrics in clearly separate fields. A
lower Dice in a DP arm versus the plain arm is the expected utility cost of privacy, not
a bug — the size of that gap (tunable via `clip_norm`/`noise_multiplier`) is the actual
privacy/utility trade-off this module makes explicit and measurable.
