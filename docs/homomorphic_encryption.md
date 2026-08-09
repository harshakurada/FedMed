# TenSEAL CKKS Homomorphic Encryption for Federated Model Updates (Module 9)

This is a simulated research/portfolio project. **It is not clinically validated, and no
claim of formal cryptographic security is made beyond what is actually implemented and
tested below.**

## Why homomorphic encryption

Modules 7–8 gave FedMed real FedAvg and real TLS/mTLS transport — but TLS only protects
data *in transit*. Once a hospital's model update reaches the aggregation server, TLS's
job is done and the server can read it in plaintext. Module 9 adds a third, independent
protection: the aggregation server combines the 3 hospitals' updates **without ever
decrypting any individual one** — only the final aggregate is ever decrypted, and by a
party distinct from the server.

## What CKKS provides

TenSEAL's CKKS scheme supports *approximate* arithmetic (addition, and multiplication by
a plaintext scalar) directly on encrypted real-valued vectors. That's exactly what
weighted FedAvg needs — `sum_i(num_examples_i * update_i)` — and nothing more; this
project never needs ciphertext-ciphertext multiplication or slot rotation, so it never
generates or transmits Galois/relinearization keys for those operations.

## Threat model

**Assumed / addressed:**
- Hospital clients should not expose their model updates in plaintext to the aggregation
  server. *(Addressed: the server only ever holds ciphertexts + a public CKKS context.)*
- The server should aggregate encrypted updates without decrypting individual
  contributions. *(Addressed and tested — see "Why the server can't decrypt" below.)*
- TLS protects the network transport (Module 8); CKKS protects the *value* while the
  server computes on it.

**Explicitly NOT claimed or addressed by this module:**
- Perfect privacy, or protection against every attack.
- Secure aggregation against a *malicious* client (a hospital that submits a
  maliciously-crafted ciphertext, or colludes with the server) — this module assumes
  honest-but-curious participants, the standard FedAvg trust assumption, not an
  adversarial one.
- Differential privacy (that's Module 10's job — no noise injection, gradient clipping,
  or privacy-budget accounting exists here).
- Complete protection against model-inversion or membership-inference attacks on the
  *final decrypted aggregate* — once decrypted, it's an ordinary model update, exactly as
  privacy-exposed as Module 7's plaintext FedAvg.
- **Ciphertext integrity/authentication.** This was discovered by testing, not assumed:
  CKKS decryption does **not** fail closed on a wrong/mismatched context — decrypting a
  ciphertext under an independently-generated context silently returns meaningless
  numbers (observed magnitude ~1e31 for inputs of order 1.0) rather than raising an
  error. TenSEAL provides no built-in ciphertext authentication. This project's own
  mitigation — a SHA-256 fingerprint of the public context, checked by the aggregation
  server before accepting any update (`server/federated/encrypted/fingerprint.py`) — is
  what actually makes "wrong context is rejected" true here; it is not a property CKKS
  gives you for free.

## Key ownership

| Component | Holds | Can decrypt? |
|---|---|---|
| Each hospital | Public CKKS context only (loaded from bytes with no secret key) | No |
| `EncryptedAggregationServer` (`server/federated/encrypted/aggregator.py`) | Public CKKS context only | No — structurally: this class has no code path that ever constructs a private context (verified in `server/tests/test_ckks_security.py`, not just documented) |
| `KeyHolder` (`server/federated/encrypted/key_holder.py`) | The private CKKS context (secret key) | **Yes — the only component in this project that ever calls `.decrypt()`** |

For this local simulation, `KeyHolder` is a distinct Python module/object from both the
hospitals and the aggregation server — modeling a party the server is not (in a real
deployment: the hospitals collectively, or an independent trusted auditor — not the
compute provider running the aggregation server). The secret key never appears in server
source code, server configuration, the Flower strategy, gRPC payloads, Git history, or
logs.

## Encryption flow

```
HospitalNode.fit()  [Module 6/7, unchanged]  ->  plaintext NDArrays + num_examples
        v
flatten_model_parameters -> chunk_values -> encrypt (hospital's own public-context copy)
        v
gRPC SubmitEncryptedUpdate  [extends Module 8's SAME mTLS service -- not a second gRPC
        v                    layer; caller identity re-verified against the already-
        v                    authenticated client certificate]
EncryptedAggregationServer.submit_update()  [validates round/model_version/context
        v                                    fingerprint/hospital before accepting]
homomorphically_add_updates()  [CKKSVector ops only: sum_i(num_examples_i * ciphertext_i)]
        v
KeyHolder.decrypt_aggregate()  [the ONLY .decrypt() call anywhere in this project;
        v                        divides by total_examples here, once, in plaintext]
unflatten_model_parameters -> reconstructed state_dict -> load_state_dict(strict=True)
```

## Homomorphic aggregation — the weighting, precisely

Clients encrypt their **raw, unweighted** flattened update. The server homomorphically
computes `sum_i(num_examples_i * ciphertext_i)` — a plaintext-scalar multiply (each
hospital's own public, non-sensitive sample count) then a ciphertext-ciphertext add,
neither of which needs Galois or relinearization keys. `KeyHolder` divides the decrypted
sum by `total_examples` **once, in plaintext, after decrypting**.

This is mathematically identical to `(N_A·A + N_B·B + N_C·C) / (N_A+N_B+N_C)` — the same
structure `flwr.server.strategy.aggregate.aggregate` uses for plaintext FedAvg
(weight-then-sum, normalize once at the end) — just with the final division deferred
past decryption instead of applied before encryption. That deferral is a deliberate
choice: dividing a decrypted value by a known public scalar is exact and free, and avoids
a second homomorphic multiply that would otherwise consume CKKS's limited multiplicative
depth for no numerical benefit.

## Why the server can't decrypt (not just "shouldn't")

`EncryptedAggregationServer` and `homomorphically_add_updates`
(`server/federated/encrypted/aggregator.py`/`encryption.py`) never import or construct a
private CKKS context anywhere — checked by a static source-inspection test
(`server/tests/test_ckks_security.py`) in addition to the obvious runtime one
(`ts.context_from(server.public_context_bytes).has_secret_key() is False`). A second test
spies on `CKKSVector.decrypt` during aggregation and asserts it is never called. This is
a structural guarantee, not a code-review convention.

## CKKS parameters and approximation

Documented development configuration (`server/federated/encrypted/ckks_config.py`,
`FEDMED_CKKS_*`-overridable): `poly_modulus_degree=8192`,
`coeff_mod_bit_sizes=[60, 40, 40, 60]`, `global_scale=2**40`. CKKS slot capacity is
`poly_modulus_degree / 2` (4096 here) — encrypting more values than that in one
`CKKSVector` triggers TenSEAL's own undocumented auto-batching fallback (which disables
further homomorphic operations); this project never relies on that and instead chunks
explicitly (`server/federated/encrypted/chunking.py`).

CKKS is **approximate**, not exact — real, observed error from this project's own
numerical-accuracy test (`server/tests/test_ckks_aggregation.py`, 3 clients, weighted
sum of random values in [-1, 1]): **max absolute error ≈ 6×10⁻⁸, mean absolute error ≈
4×10⁻⁹, relative error ≈ 8×10⁻⁷** — many orders of magnitude below the configured
tolerance (`numerical_tolerance`, default `1e-3`). Nothing rounds weights aggressively to
hide error; the actual measured numbers are asserted against the tolerance and printed by
the test.

## TLS vs. CKKS — neither replaces the other

**TLS (Module 8)** protects data while it moves across the network — without it, anyone
observing the connection could read the raw bytes in flight, encrypted or not. **CKKS**
protects the *value* while the server computes on it — without it, the server (even over
a perfectly secure TLS connection) would receive each hospital's update in cleartext.
Module 9's gRPC transport (`SubmitEncryptedUpdate`) uses **both**: it's the same
mTLS-secured channel Module 8 built, now carrying ciphertext payloads instead of a health
ping. Removing either layer weakens the system in a different way — TLS's absence exposes
transport; CKKS's absence exposes the server-side aggregation step.

## Performance (measured in this environment, not claimed from documentation)

3-hospital smoke test (tiny synthetic data, `chunk_size=64`):
encryption ≈ 1.3s total (3 hospitals), homomorphic aggregation ≈ 0.35s, decryption ≈
0.05s, public context ≈ 465 KB (with `save_galois_keys=False` — the same context
serialized with default settings was **35 MB**, since this project never needs Galois
keys for +/scalar-* only).

Real FedMed 3D U-Net (Module 4's actual architecture, 4,810,074 parameters, 1175 chunks
at the default `chunk_size=4096`, measured in `server/tests/test_ckks_model_compatibility.py`):
encryption ≈ **8.2s**, decryption ≈ **2.6s**, for one hospital's one update. This scales
with parameter count and chunk count — a full multi-round encrypted training experiment
across all 3 hospitals was deliberately not run automatically (the task's explicit smoke-
test scope); the numbers above are what such a run would actually cost per
update/round, measured, not estimated.

## Plaintext vs. encrypted FedAvg — both remain available

Module 7's plaintext path (`server/federated/experiment.py`,
`run_federated_experiment`) is **completely untouched** — nothing in Module 9 modifies
it. Module 9 is entirely additive (`server/federated/encrypted/`, a new subpackage) and
is invoked separately:

```powershell
# Plaintext FedAvg (Module 7, unchanged)
.venv\Scripts\python.exe -m server.federated.run_experiment

# Encrypted FedAvg (Module 9) -- single round, real 3D U-Net by default
.venv\Scripts\python.exe -m server.federated.encrypted.run_encrypted_round
```

There is no flag inside `experiment.py` that silently switches behavior — the two paths
are separate modules, so choosing one never risks regressing the other.

## How to run the tests

```powershell
.venv\Scripts\activate
.venv\Scripts\python.exe -c "import tenseal; print(tenseal.__version__)"   # verify TenSEAL is installed
.venv\Scripts\python.exe -m pytest server\tests -k ckks -v                 # Module 9 tests only
.venv\Scripts\python.exe -m pytest server\tests\test_encrypted_smoke.py -v # 3-hospital smoke test
.venv\Scripts\python.exe -m pytest -q                                      # everything, Modules 1-9
```

## How to run encrypted FedAvg yourself

```powershell
$env:FEDMED_BRATS_ROOT = "C:\path\to\your\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
.venv\Scripts\python.exe -m server.federated.encrypted.run_encrypted_round
```

Configurable via `FEDMED_CKKS_*` environment variables (`server/federated/encrypted/ckks_config.py`).
Does **not** run automatically as a side effect of anything else in this project.

## Security limitations

This module provides real CKKS encryption of model updates and a real, tested guarantee
that the aggregation server never decrypts an individual hospital's contribution. It does
**not** provide: differential privacy, secure aggregation against a malicious
participant, ciphertext authentication beyond this project's own context-fingerprint
check, or any protection for the model once it is legitimately decrypted by `KeyHolder`.
See "Threat model" above for the complete list. This is a research/portfolio
implementation, not a certified or audited cryptographic system.
