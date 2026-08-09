# Secure gRPC Communication + TLS + Node Resilience (Module 8)

This is a simulated research/portfolio project running on one local machine. **TLS here
means encrypted, authenticated network transport for a local simulation — it is not a
production healthcare deployment, and it is not the same thing as the homomorphic
encryption / differential privacy work planned for later modules** (see "Security
limitations" below).

## Why gRPC, and why two separate gRPC surfaces

Module 7 built a real FedAvg round but ran it in-process, since gRPC/TLS were explicitly
out of scope then. This module lifts that boundary — but honestly, not by inventing a
second federated-learning transport. Inspecting the installed Flower 1.33 CLI directly
(`flower-superlink --help`, `flower-supernode --help`) shows Flower's own deployment
engine (SuperLink + SuperNode) **is already real gRPC**, and **already supports real
TLS** natively (`--ssl-certfile`/`--ssl-keyfile`/`--ssl-ca-certfile` on the SuperLink,
`--root-certificates` on the SuperNode; HTTPS is the *default* — `--insecure` is required
to turn it off). So:

1. **FedAvg's actual model-parameter/metric traffic uses Flower's own gRPC**, now
   configured with real TLS. Flower remains 100% responsible for federated learning —
   nothing about `FedAvg`, `configure_fit`, or aggregation was touched.
2. **A second, small, genuinely-new gRPC service** (`server/federated/grpc_service/`)
   covers something that doesn't exist anywhere else: a hospital proving its identity
   over a *mutually*-authenticated channel. Flower's Fleet API only supports one-way TLS
   (the SuperNode verifies the SuperLink's certificate; there is no client-certificate
   option) — its separate node-authentication mechanism uses raw keypairs, not X.509
   client certificates. So genuine mutual TLS, with a hospital's identity tied to its
   certificate, is only possible in this small custom service. It carries exactly one
   RPC (`HealthCheck`) and never carries model parameters, training data, or labels.

Module 7's in-process orchestrator (`server/federated/experiment.py`) is still how the
round loop itself is built, tested, and proven — it isn't replaced by either gRPC surface
above; see `docs/federated_training.md`.

## Why TLS

Without TLS, gRPC traffic between a hospital and the server is plaintext on the network —
anyone who can observe the connection can read model weight updates and metrics in
transit. TLS encrypts that traffic and lets each side verify who it's actually talking to.

## Certificate architecture

A single local development CA signs a server certificate and one client certificate per
hospital:

```
certs/
├── ca.crt / ca.key           # Dev CA -- ca.key must NEVER be committed
├── server.crt / server.key   # SAN: localhost, 127.0.0.1 -- used by the coordination
│                              # service (and can be reused for the SuperLink)
├── hospital_a.crt / hospital_a.key   # CN=hospital_a
├── hospital_b.crt / hospital_b.key   # CN=hospital_b
└── hospital_c.crt / hospital_c.key   # CN=hospital_c
```

`server/federated/grpc_service/health_server.py`'s `FedMedHealthServicer` reads a
connecting hospital's identity from the **verified certificate's Common Name**
(`context.auth_context()['x509_common_name']`), not from the request body — a request
claiming to be `hospital_a` over a connection authenticated as `hospital_b` is rejected
(`PERMISSION_DENIED`), proven by
`server/tests/test_grpc_tls.py::test_identity_claimed_in_request_must_match_the_certificate`.

The entire `certs/` directory is gitignored (`.gitignore`; `*.pem`/`*.key`/`*.crt` were
already covered, `certs/` was added explicitly) — these are machine-generated, regenerable
dev artifacts, never committed.

## Certificate generation (PowerShell)

Requires `openssl` on PATH (Git for Windows bundles one). The script fails clearly,
without installing anything, if it isn't found.

```powershell
.venv\Scripts\python.exe scripts\generate_dev_certs.py
# Regenerate (invalidates certs any running server/client currently trusts):
.venv\Scripts\python.exe scripts\generate_dev_certs.py --force
```

## Starting the secure system

**The FedMed mTLS coordination service** (proves hospitals can reach the server over an
authenticated channel — the piece this module's automated tests exercise):

```powershell
$env:FEDMED_GRPC_CA_CERT = "$PWD\certs\ca.crt"
$env:FEDMED_GRPC_SERVER_CERT = "$PWD\certs\server.crt"
$env:FEDMED_GRPC_SERVER_KEY = "$PWD\certs\server.key"
.venv\Scripts\python.exe -c "from server.federated.grpc_service.config import DEFAULT_CONFIG; from server.federated.grpc_service.health_server import create_grpc_server; s = create_grpc_server(DEFAULT_CONFIG); s.start(); print('listening on', DEFAULT_CONFIG.address); s.wait_for_termination()"
```

**The real Flower live deployment** (Flower's own gRPC+TLS carrying FedAvg traffic) —
manually verified: all 3 SuperNodes registered with the SuperLink over a genuine TLS
handshake (unique SuperNode IDs assigned, `Fleet.ActivateNode`/`Fleet.PullMessages`
logged server-side for each) using `--isolation process` (the default `--isolation
subprocess` fails to start at all on this Windows/Python 3.14 environment with
`[WinError 2] The system cannot find the file specified`, independent of TLS — reproduced
even with `--insecure`, so it's an environment/packaging issue, not a Module 8 bug):

```powershell
# 1. SuperLink, real TLS (HTTPS is the default; --insecure would disable it)
flower-superlink --ssl-ca-certfile certs\ca.crt --ssl-certfile certs\server.crt --ssl-keyfile certs\server.key --isolation process

# 2. One SuperNode per hospital, in separate terminals -- each verifies the SuperLink's
#    certificate via --root-certificates (one-way TLS; Flower's Fleet API has no
#    client-certificate option, see "Why gRPC" above). Needs FEDMED_BRATS_ROOT set in
#    each terminal, same as any other real-data run.
flower-supernode --root-certificates certs\ca.crt --superlink 127.0.0.1:9092 --node-config "partition-id=0" --clientappio-api-address 127.0.0.1:9095 --isolation process
flower-supernode --root-certificates certs\ca.crt --superlink 127.0.0.1:9092 --node-config "partition-id=1" --clientappio-api-address 127.0.0.1:9096 --isolation process
flower-supernode --root-certificates certs\ca.crt --superlink 127.0.0.1:9092 --node-config "partition-id=2" --clientappio-api-address 127.0.0.1:9097 --isolation process

# 3. Submit the run -- root-certificates must be an absolute path (Flower requirement),
#    so it's passed here rather than hard-coded anywhere committed. The first `flwr run`
#    against a federation auto-migrates pyproject.toml's [tool.flwr.federations] into
#    ~/.flwr/config.toml (Flower's newer connection-config location) -- expected, one-time.
flwr run . local-deployment --federation-config "root-certificates=`"$PWD\certs\ca.crt`""
```

**Known limitation, reported honestly:** SuperNode↔SuperLink TLS registration was
verified working end-to-end as above. Actually *submitting* a run this way
(`flwr run . local-deployment`) additionally hit `Connection to the SuperLink is
unavailable` from the CLI's run-submission client in this environment, despite the
SuperLink being confirmed listening and the SuperNodes already TLS-registered against it
— this looks like a separate Flower-CLI/Windows/Python-3.14 issue in the run-submission
path (TCP connections from the CLI to the Control API were observed completing at the
socket level, so this is not simply "wrong address"), not a certificate/TLS
misconfiguration. This is flagged as a follow-up rather than claimed as working — the
automated test suite (`server/tests/test_grpc_tls.py`, 8 tests against a real running
mTLS server) is the reliable, repeatable proof that this project's TLS/mTLS
implementation itself is correct.

## Node failure and reconnection behavior

Handled at the Flower/FedAvg layer (`server/federated/experiment.py`), since that's where
Flower's own min-clients/failure-acceptance mechanism actually lives — TLS doesn't provide
this.

- Each hospital's `fit()` call is wrapped individually. If it raises (a dropped
  connection, in a live deployment), the round is **not** aborted — the hospital is
  recorded in that round's `failed_hospital_ids` and simply excluded from aggregation.
  **No fake or substitute parameters are ever used for a missing hospital** — FedAvg
  aggregates only from whoever actually succeeded (`flwr.server.strategy.FedAvg`'s own
  `accept_failures=True` default behavior, unmodified).
- If every sampled hospital fails, the round raises a clear `RuntimeError` naming how
  many failed — it never hangs and never fabricates a result.
- **Reconnection**: every round's `FitIns` carries the full current global parameters
  (Flower's own design, not something added here), so a hospital that failed round N and
  succeeds in round N+1 trains from the correct, current global model automatically —
  nothing needs to detect or repair "staleness" in its starting point.
- **Stale-update protection**: `on_fit_config_fn` (`server/federated/strategy.py`) tells
  every hospital which round it's training for; `HospitalNodeClient.fit` echoes that
  round number back in its metrics. If a response's echoed round doesn't match the
  current round, `experiment.py` excludes it from aggregation and records it in
  `stale_hospital_ids` — a late/delayed response can never be applied to a newer global
  model.

Verified end-to-end in `server/tests/test_federated_resilience.py`: one hospital dropped
mid-round (round completes on the other two, recorded, no fake parameters), the same
hospital fully reconnecting next round, a fabricated stale response being excluded, and a
total-outage case raising clearly rather than hanging.

`FederatedConfig.round_timeout_seconds` (`FEDMED_FED_ROUND_TIMEOUT`) is wired into the
*live* deployment's `ServerConfig(round_timeout=...)` (confirmed to exist on the installed
Flower version) so one unreachable hospital can't block the live server indefinitely. It
is **not** meaningful for the in-process orchestrator — that code path calls each
hospital synchronously in-process (a plain Python call, not a network round-trip), so
there's no wall-clock wait to bound there; a dropped hospital is instead handled by the
try/except above. This distinction is documented rather than faked.

## Security limitations

Current (Module 8): **encrypted, authenticated network transport**. Flower's live
deployment uses real TLS (server-authenticated); the FedMed coordination service uses
real mutual TLS (both sides authenticated, hospital identity tied to its certificate).

**Not yet implemented:**
- Homomorphic encryption (TenSEAL) or any encrypted aggregation.
- Differential privacy.
- Secure aggregation protocols.
- Protection against a malicious server: TLS protects data *in transit*; the server still
  receives each hospital's model update in plaintext once decrypted at the transport
  layer. A server operator can inspect those updates. This system is **not** currently
  privacy-preserving against an untrusted server — that's later modules' work.

## Troubleshooting common TLS errors

| Symptom | Cause | Fix |
|---|---|---|
| `openssl: command not found` when generating certs | OpenSSL isn't on PATH | Install Git for Windows (bundles OpenSSL) or OpenSSL directly, then re-open the terminal |
| `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate` | Client's CA doesn't match the server's certificate's issuer | Regenerate both from the *same* CA (`generate_dev_certs.py`), or point `FEDMED_GRPC_CA_CERT`/`--root-certificates` at the right `ca.crt` |
| `PEER_DID_NOT_RETURN_A_CERTIFICATE` | Client connected without presenting a certificate to an mTLS-only server | Use `create_secure_channel` (always presents the hospital's client cert), not a bare `grpc.ssl_channel_credentials()` without `private_key`/`certificate_chain` |
| `WRONG_VERSION_NUMBER` | A plaintext/insecure channel was used against a TLS-only server | Use `create_secure_channel`/`flower-supernode` (drop `--insecure`), not `grpc.insecure_channel` |
| `root-certificates` rejected as "expected absolute path" by `flwr run` | Flower requires an absolute path for this field, and it must not be hard-coded (machine-specific) into `pyproject.toml` | Pass it via `--federation-config "root-certificates=\"<absolute path>\""` at invocation time, as shown above |
| `PERMISSION_DENIED: hospital_id ... does not match the authenticated certificate` | A `HealthCheck` request claimed an identity different from the certificate that authenticated the connection | Use the matching hospital's own certificate for its own identity claim |
| `flower-superlink` exits immediately with `Exit Code: 104` / `[WinError 2] The system cannot find the file specified` | The default `--isolation subprocess` mode fails to spawn on this Windows/Python 3.14 setup — reproduced even with `--insecure`, so it's unrelated to certificates | Add `--isolation process` |
| `'charmap' codec can't encode character ...` from `flwr` CLI commands | Windows console codepage can't print one of Flower's Unicode log characters | `$env:PYTHONUTF8 = "1"` before running `flwr` commands |
| `flwr run` reports `Connection to the SuperLink is unavailable` even though the SuperLink is confirmed listening and SuperNodes already TLS-registered against it | Not reproduced as a certificate/TLS problem in testing (SuperNode↔SuperLink TLS registration worked; only the separate run-submission step failed) — appears to be a Flower-CLI/Windows/Python-3.14 issue in this environment | Unresolved here; treat the automated `server/tests/test_grpc_tls.py` suite as the authoritative proof of this project's TLS/mTLS correctness, and retry `flwr run` on a Linux/macOS dev machine or an older Python if you need the full live pipeline |
