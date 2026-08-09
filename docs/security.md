# Consolidated Security Model (Module 12)

This is a simulated research/portfolio project. **No claim of formal cryptographic
certification, penetration testing, or production-grade security is made.** See the
disclaimer at the bottom of this page and in the main `README.md`.

This page consolidates the threat model, security mechanisms, and key-ownership facts
already documented per-module (`docs/secure_communication.md`, `docs/homomorphic_encryption.md`,
`docs/differential_privacy.md`, `docs/dashboard.md`) into one place, plus the Module 12
audit findings. It does not repeat those docs' full detail — follow the links for that.

## Threat model, combined

**Trust assumption throughout: honest-but-curious.** Every hospital and the server follow
the protocol correctly (train on their own data, submit what the protocol says to submit,
aggregate correctly); the concern is what an honest-but-curious *observer* of network
traffic or server-side state could learn, not a malicious participant submitting crafted
data or deviating from the protocol. This is the standard FedAvg trust assumption, stated
explicitly rather than left implicit, in `docs/homomorphic_encryption.md` and
`docs/differential_privacy.md`.

**What is addressed, layer by layer:**

| Layer | Protects against | Does NOT protect against |
|---|---|---|
| mTLS (Module 8) | Network eavesdropping/tampering in transit; unauthenticated connections (server verifies hospital certs, hospital verifies server cert) | Anything once data is received in plaintext at either endpoint |
| CKKS (Module 9) | The server operator reading an individual hospital's update | A malicious/colluding client; what the final *decrypted* aggregate reveals once legitimately decrypted |
| Differential Privacy (Module 10) | What one hospital's *own contribution this round* could reveal about its aggregate training signal, at the **client (hospital) level** | Patient-level privacy within a hospital's own data; membership inference on the final trained model after many rounds (only single-round contribution is bounded per round) |
| Dashboard payload allowlist (Module 11) | Any raw data, model tensor, or secret ever reaching a WebSocket message, even by accident (enforced at construction, not by convention) | N/A — this is a data-minimization control, not a confidentiality mechanism against a network attacker (the dashboard itself has no TLS or auth — see below) |

**Explicitly NOT addressed anywhere in this project:**
- Malicious/Byzantine clients (a hospital submitting a crafted update to corrupt the
  global model, or attempting a model-inversion attack against other hospitals via
  crafted gradients).
- A malicious/compromised aggregation server deviating from the protocol (e.g. one that
  runs different code than what's in this repo).
- Ciphertext authentication beyond this project's own SHA-256 context-fingerprint check
  (`server/federated/encrypted/fingerprint.py`) — CKKS itself has no built-in ciphertext
  integrity/authentication, confirmed by testing (a wrong-context decrypt silently
  returns garbage rather than raising).
- Dashboard authentication or transport encryption — `server/dashboard/websocket_server.py`
  accepts any local `ws://` connection; explicitly a local development monitoring
  interface (see `docs/dashboard.md`'s "Authentication" section).
- HIPAA/GDPR compliance, clinical validation, or any regulatory certification of any kind.

## Key ownership (authoritative table)

| Key/secret | Held by | Never held by |
|---|---|---|
| CKKS secret key | `KeyHolder` only (`server/federated/encrypted/key_holder.py`) | Hospitals, `EncryptedAggregationServer`, the gRPC coordination service, the dashboard |
| gRPC CA private key (`ca.key`) | Whoever ran `generate_dev_certificates` locally; gitignored, never committed | Any tracked file in this repository |
| gRPC server private key | The coordination service process | Hospitals, `KeyHolder` |
| Each hospital's gRPC client private key | That hospital's own process only | Any other hospital, the server |
| DP Gaussian noise RNG seed | Never fixed in production (`np.random.default_rng()` unseeded); a seed is only ever used in tests, explicitly for reproducibility | The dashboard, any logged output, any WebSocket payload (`dp_seed` is on the dashboard's forbidden-field list) |
| Raw patient data (images, labels, patient IDs) | Each hospital's own local filesystem only | The server, any other hospital, any gRPC payload (only CKKS ciphertext + approved metadata cross the wire), the dashboard |

## Test evidence — which test proves which property

| Property | Test file |
|---|---|
| Server never constructs a CKKS private context / never calls `.decrypt()` | `server/tests/test_ckks_security.py` |
| Wrong-context CKKS update is rejected (fingerprint check) | `server/tests/test_ckks_security.py::test_wrong_context_update_is_rejected_via_fingerprint_check` |
| Stale round / duplicate update / wrong model version rejected | `server/tests/test_ckks_security.py` |
| mTLS: client without a certificate is rejected | `server/tests/test_grpc_tls.py` |
| mTLS: identity claimed in a request must match the authenticated certificate | `server/tests/test_grpc_tls.py`, `server/tests/test_ckks_grpc.py::test_identity_mismatch_over_grpc_is_rejected` |
| Plain insecure channel rejected by the TLS-only server | `server/tests/test_grpc_tls.py::test_plain_insecure_channel_is_rejected_by_the_tls_only_server` |
| DP update actually differs from the raw post-training parameters (mechanism really ran) | `server/tests/test_dp_three_hospital_smoke.py`, `server/tests/test_final_integration.py` |
| Epsilon is computed from the configured mechanism, never fabricated | `server/tests/test_dp_accountant.py` |
| Cumulative privacy budget never resets between rounds | `server/tests/test_dp_accountant.py` |
| No raw MRI/patient/tensor data reaches the DP or CKKS layer | `server/tests/test_dp_three_hospital_smoke.py`, `server/tests/test_ckks_security.py` |
| Dashboard payload construction rejects any forbidden field | `server/tests/test_dashboard_events.py` |
| No forbidden field reaches a live WebSocket message end-to-end | `server/tests/test_dashboard_experiment_integration.py`, `server/tests/test_final_integration.py` |
| Dashboard module never imports TenSEAL / never touches a TLS key / never calls the DP mechanism | `server/tests/test_dashboard_module_compatibility.py` |
| A dropped hospital never blocks a round or receives fake substitute parameters | `server/tests/test_federated_resilience.py`, `server/tests/test_final_integration.py::test_hospital_failure_during_integrated_round_still_completes_with_the_rest` |
| DP + CKKS + real mTLS + dashboard all genuinely compose in one round | `server/tests/test_final_integration.py` (Module 12) |

## Module 12 security audit (performed this module, read-only)

Findings, from actually running the checks against the tracked repository (not assumed):

- `git ls-files` for `.key`/`.pem`/`.crt`/secret-shaped filenames: **none tracked.**
- `password=`/`secret=`/`api_key=`-style patterns across tracked `.py`/`.js`/`.jsx`:
  **none found.**
- Hardcoded absolute local paths (e.g. this machine's home directory) in tracked `.py`
  files: **none found** — every path-like config field is a dataclass default
  (`./data/...`, `./certs/...`, `./checkpoints/...`) overridable via a `FEDMED_*`
  environment variable, never a machine-specific absolute path baked into source.
- `secret_key`/`private_key`/`.decrypt(`/`patient_id`/`patient_name` matches across the
  tracked tree: every match is the legitimate, already-tested security-boundary code
  itself (`key_holder.py` constructing the one authorized private context,
  `aggregator.py`'s fingerprint check, `health_client.py` passing a TLS client key to
  `grpc.ssl_channel_credentials`) or a test file that verifies that boundary — no leak,
  no bare `print(model)` / `print(state_dict)` / secret-logging anywhere.
- `requirements.txt` / `.gitignore`: reviewed in full — no duplicate or conflicting
  pinned dependency, `.gitignore` already covers `certs/`, `*.key`/`*.pem`/`*.crt`,
  `checkpoints/`, `node_modules/`, `.venv/`. No changes were needed to either file.
- `server/tests/test_final_integration.py` (Module 12, new): confirmed structurally, via
  an instance-level check (`ts.context_from(aggregation_server.public_context_bytes)
  .has_secret_key() is False`) and `not hasattr(aggregation_server, "_context")`, that
  the real integration round's server-side aggregation object never holds a CKKS secret
  key — the same guarantee `test_ckks_security.py` proves in isolation, now re-confirmed
  inside the full composed round.

## Known limitations (repeated from per-module docs, consolidated)

- Basic sequential composition for cumulative DP epsilon is conservative, not tight — a
  real RDP/moments accountant would report a lower (better) cumulative epsilon for the
  same mechanism; this project deliberately did not implement one from scratch rather
  than risk an incorrect "tighter" number (`docs/differential_privacy.md`).
- CKKS ciphertext integrity relies entirely on this project's own context-fingerprint
  check, not a property of CKKS itself (`docs/homomorphic_encryption.md`).
- Flower's live SuperLink/SuperNode deployment only supports one-way TLS; genuine mutual
  TLS exists only in this project's own small coordination service
  (`docs/secure_communication.md`).
- The dashboard has no authentication or transport encryption; it is explicitly a local
  development monitoring interface (`docs/dashboard.md`).
- `flwr run` against the live SuperLink/SuperNode deployment hit an unresolved
  environment-specific connection issue on this Windows/Python 3.14 setup, reported
  honestly rather than worked around by weakening a claim (`docs/secure_communication.md`).

## Disclaimer

FedMed is a research/portfolio implementation of federated learning, homomorphic
encryption, differential privacy, and secure transport concepts. **It has not undergone
independent security review, penetration testing, or formal cryptographic audit. It makes
no HIPAA, GDPR, or other regulatory compliance claim, and no claim of clinical validation.**
Do not use it, as-is, to process real patient data.
