# Security Checklist (Module 13)

A pre-publish checklist for this repository. Each item links to the check that actually
verified it (a test, or a command run against the tracked tree) — nothing here is
checked off on assumption. Re-run the referenced commands yourself before trusting this
list; it reflects the state as of 2026-08-09 (Module 13).

- [x] **TLS enabled** — real mutual TLS in FedMed's own coordination service
      (`server/tests/test_grpc_tls.py`, 8 tests against a real running mTLS server);
      real server-authenticated TLS in Flower's own live deployment
      (`docs/secure_communication.md`).
- [x] **Private keys excluded from version control** — `.gitignore` covers `*.pem`,
      `*.key`, `*.crt`, `certs/`; verified via `git ls-files | grep -iE
      "\.(key|pem|crt)$"` returning nothing.
- [x] **CKKS secret key protected** — exists only inside `KeyHolder`; the aggregation
      server and every hospital hold only a public context; verified structurally
      (`server/tests/test_ckks_security.py`) and re-confirmed inside the full composed
      round (`server/tests/test_final_integration.py`).
- [x] **No patient data in dashboard** — payload allowlist rejects `patient_id`,
      `patient_name`, `MRI`, `image_data`, `mask` at event construction
      (`server/tests/test_dashboard_events.py`).
- [x] **No raw model update transmitted** — every gRPC payload carrying a model update
      is CKKS ciphertext bytes; the plaintext path (`server/federated/experiment.py`)
      transmits only in-process, never over a network in this project.
- [x] **DP enabled when required** — `DPConfig.enabled` is explicit and defaults to
      `False`; every experiment/test that claims DP protection passes
      `enabled=True` explicitly, never assumed.
- [x] **Privacy accountant active when DP is enabled** — `PrivacyAccountant.record_round`
      is called for every DP-protected update; cumulative epsilon never resets between
      rounds (`server/tests/test_dp_accountant.py`).
- [x] **Sensitive logs removed** — no `print(model)` / `print(state_dict)` / secret-value
      logging anywhere in the tracked tree (checked via `git grep` across `.py` files,
      Module 12/13 audits).
- [x] **`.gitignore` verified** — covers certs/keys, `checkpoints/`, `data/`,
      `dashboard/node_modules/`, `.venv/`, `*.env`; reviewed in full, Module 12.
- [x] **No secrets committed** — `git ls-files` shows no `.env`, no cert/key file, no
      hardcoded `password=`/`secret=`/`api_key=` pattern in any tracked `.py`/`.js`/`.jsx`
      file (Module 12 audit, re-checked Module 13 for the files this module added).
- [x] **Dashboard payload audit passed** — `server/tests/test_dashboard_events.py`
      checks every explicitly-forbidden field directly; live end-to-end re-confirmation
      in `server/tests/test_dashboard_experiment_integration.py` and
      `server/tests/test_final_integration.py`.

## How to re-verify this checklist yourself

```powershell
# Private keys / certs never tracked
git ls-files | Select-String -Pattern "\.(key|pem|crt)$"

# No hardcoded secret-shaped assignment
git grep -inE "(password|secret|api_key)\s*="  -- "*.py" "*.js" "*.jsx"

# Full test suite, including the security-specific tests
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m pytest server\tests -k "security or dashboard_events or grpc_tls" -v
```

None of these were skipped or assumed for this checklist — see
`docs/final_validation_report.md` for the actual command output from the run that backs
it.
