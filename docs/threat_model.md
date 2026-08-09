# Threat Model (Module 13)

This is a simulated research/portfolio project. **No claim of formal threat-modeling
certification is made** — this is a plain-language threat/mitigation table for portfolio
and interview purposes, consolidating what's already proven (with tests) in
`docs/security.md`. Trust assumption throughout: **honest-but-curious** (every hospital
and the server follow the protocol; the concern is what an observer could learn, not a
participant deviating from the protocol — see `docs/security.md` for what this
explicitly does *not* cover).

| Threat | Mitigation | Evidence |
|---|---|---|
| Raw medical data exposure (a hospital's MRI scans or labels leaking to the server or another hospital) | Data remains local — only model parameters/updates ever leave a hospital, and only after DP + CKKS are applied to them | `server/tests/test_dp_three_hospital_smoke.py`, `hospital_nodes/node.py`'s own design (raw `StudyRecord`s are a private attribute, never returned by any public method) |
| Network interception (an observer reading traffic between a hospital and the server) | TLS (real mutual TLS in FedMed's own coordination service; real server-authenticated TLS in Flower's live deployment) | `server/tests/test_grpc_tls.py` (8 tests against a real running mTLS server) |
| Server seeing a raw (unencrypted) client update | CKKS homomorphic encryption — the server aggregates ciphertexts and never decrypts an individual update | `server/tests/test_ckks_security.py` (static + runtime check that the server-side code never constructs a private context) |
| Model-update leakage (what one hospital's contribution reveals about its own training signal) | Differential Privacy — clip + calibrated Gaussian noise on each hospital's own per-round contribution, applied *before* encryption | `server/tests/test_dp_accountant.py`, `server/tests/test_dp_three_hospital_smoke.py` |
| Secret-key compromise (CKKS private key, TLS private keys) | Key isolation: the CKKS secret key exists only in `KeyHolder`, structurally never reachable from the server or hospital code paths; TLS private keys are gitignored, never committed, and each hospital only ever holds its own | `server/tests/test_ckks_security.py`; `.gitignore`'s `*.key`/`certs/` entries; `docs/security.md`'s key-ownership table |
| A forbidden field (patient ID, MRI data, secret key, raw ciphertext) reaching the dashboard | An explicit payload allowlist, enforced at event construction, not filtered later | `server/tests/test_dashboard_events.py`, `server/tests/test_final_integration.py` |
| A dropped/failed hospital corrupting or blocking a round | The round completes with the remaining hospitals; a failed hospital never gets fake substitute parameters | `server/tests/test_federated_resilience.py`, `server/tests/test_final_integration.py::test_hospital_failure_during_integrated_round_still_completes_with_the_rest` |
| A CKKS ciphertext decrypted under the wrong context (CKKS itself doesn't fail closed on this — it silently returns garbage) | A SHA-256 context-fingerprint check, rejecting any update encrypted under a different context before it reaches aggregation | `server/tests/test_ckks_security.py::test_wrong_context_update_is_rejected_via_fingerprint_check` |

## Explicitly out of scope (limitations, not oversights)

- **Malicious/Byzantine clients** — a hospital submitting a crafted or poisoned update.
  Not defended against; the trust model is honest-but-curious.
- **A malicious or compromised server** — one that runs different code than what's in
  this repository. Not defended against.
- **Ciphertext authentication beyond the context-fingerprint check** — CKKS has no
  built-in ciphertext integrity mechanism; this project's own fingerprint check is the
  only mitigation, not a property of CKKS itself.
- **Dashboard authentication or transport encryption** — the dashboard WebSocket server
  accepts any local connection; explicitly a local development monitoring interface.
- **Patient-level privacy within a single hospital's own data** — DP here is
  client-level (hospital-level); see `docs/differential_privacy.md` for why.
- **Regulatory compliance of any kind** (HIPAA, GDPR, or otherwise) and **clinical
  validation**.

Full detail, key-ownership table, and the complete test-to-property mapping:
[`docs/security.md`](security.md).
