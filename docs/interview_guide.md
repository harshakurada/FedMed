# Interview Guide (Module 13)

Concise answers grounded in FedMed's actual implementation — not generic textbook
answers. Where relevant, each answer names the file/module that backs it.

**1. What problem does FedMed solve?**
Hospitals hold valuable medical imaging data but can't centralize it — privacy law
(HIPAA/GDPR) and patient trust mean raw MRI scans and labels shouldn't leave the hospital
that holds them. FedMed lets simulated hospitals collaboratively train a brain tumor
segmentation model without ever sharing raw data.

**2. Why federated learning?**
It keeps raw data at its source — only model updates move, and in FedMed those updates
are further protected by DP (before leaving the hospital) and CKKS encryption (in
transit and during aggregation), so even the update itself is never seen in the clear by
the server.

**3. Why not centralize the data?**
Legal/regulatory constraints and patient trust aside, it also doesn't reflect how real
hospitals operate — most won't or can't export raw patient imaging data to a third
party's server, federated or not.

**4. Why Flower?**
It's a mature, actively-maintained FL framework with a real `FedAvg` implementation and
a real gRPC/TLS live-deployment engine already built in (`server/federated/experiment.py`,
`docs/secure_communication.md`) — no need to reinvent orchestration or transport.

**5. Why FedAvg?**
The standard, well-understood aggregation baseline — weighted by each hospital's sample
count (`flwr.server.strategy.aggregate.aggregate`). Using the standard algorithm makes
the actual novelty (DP + CKKS layered on top) the thing under test, not the aggregation
math itself.

**6. Why Differential Privacy?**
CKKS hides an update's value from the server during aggregation, but says nothing about
what the *legitimately decrypted* final aggregate could reveal about a hospital's
training signal. DP bounds that separately, at the point each hospital's contribution is
produced (`server/federated/dp/`).

**7. What exactly is clipped?**
The hospital's own *delta* for the round — `post_training_params - pre_round_global_params`
— not the raw model weights (`server/federated/dp/dp_update.py::apply_dp_mechanism`).
Clipping bounds the L2 norm of that delta before noise is added.

**8. What is epsilon?**
The privacy-loss bound from the classical Gaussian mechanism:
`epsilon = sqrt(2 * ln(1.25/delta)) / noise_multiplier`
(`server/federated/dp/accountant.py::compute_epsilon`). Smaller epsilon means a stronger
privacy guarantee (more noise relative to the clip bound).

**9. What is delta?**
The (small) probability the `(epsilon, delta)`-DP guarantee could fail to hold — by
convention, much smaller than 1 over the dataset size. FedMed's default is `1e-5`.

**10. Why CKKS?**
FedAvg's aggregation only needs addition and multiplication-by-a-plaintext-scalar on
real-valued vectors (`sum_i(num_examples_i * update_i)`). CKKS is the homomorphic scheme
built for exactly that — approximate arithmetic on encrypted real numbers — without the
complexity of exact-integer schemes that don't natively fit floating-point model weights.

**11. Why TenSEAL?**
A maintained, documented Python wrapper around Microsoft SEAL implementing CKKS —
avoids hand-rolling a homomorphic-encryption library from scratch, which would be both
risky and out of scope for this project.

**12. Why TLS if CKKS already exists?**
Different layers, proven separately in `docs/homomorphic_encryption.md`'s "TLS vs. CKKS"
section: TLS protects data (and connection identity) in transit over the network; CKKS
protects the *value* from the server's own computation once received. Removing either
weakens a different part of the system — TLS's absence exposes the transport even though
the payload is ciphertext; CKKS's absence would expose the value to the server even over
a perfectly secure TLS connection.

**13. What does CKKS protect?**
The individual hospital's model-update *value* from the aggregation server's own
operator — the server computes the weighted sum on ciphertext and never decrypts any
individual contribution (structurally guaranteed, `server/tests/test_ckks_security.py`).

**14. What does TLS protect?**
Data in transit across the network, plus — via genuine mutual TLS in FedMed's own
coordination service — each side's identity (a hospital's identity is read from its
*verified certificate*, never trusted from the request body).

**15. Why use both DP and encryption?**
They protect different things at different points: CKKS protects the update while it's
in the server's hands (from the server itself); DP protects what the update *would
reveal even if the server were fully trusted or the aggregate is later, legitimately,
decrypted*. Neither substitutes for the other (`docs/security.md`'s layering table).

**16. What happens if one hospital goes offline?**
The round proceeds with the remaining hospitals — a dropped hospital never blocks the
round and never receives fake substitute parameters (`server/federated/experiment.py`,
Module 8's resilience path, `server/tests/test_federated_resilience.py`). If it
reconnects on a later round, it trains from the correct current global model, since every
round's `FitIns` carries the full current parameters regardless of who missed the last
round.

**17. How are hospital datasets separated?**
Patient-level partitioning, not per-slice or per-image: the global train/validation
split happens once (Module 3/5), then the training studies are partitioned across the 3
hospitals such that no single patient/study is ever split across two hospitals — verified
by `hospital_nodes/partition.py::verify_partition_isolation` and its own test suite.

**18. Why 3D U-Net?**
A well-established encoder-decoder architecture for volumetric (3D) medical image
segmentation — the standard reference architecture for tasks like BraTS, and MONAI's own
recommended starting point.

**19. Why MONAI?**
A PyTorch-based framework purpose-built for medical imaging: 3D-aware transforms, Dice
loss/metrics, and NIfTI I/O out of the box, rather than re-implementing those from
generic vision libraries not designed for volumetric medical data.

**20. Why WebSockets?**
A real-time, low-overhead, server-push protocol — the dashboard needs to reflect live
training/round/privacy/security events without polling. Python's `websockets` library
was already an approved dependency; no new framework (Socket.IO, etc.) was added.

**21. What does the dashboard monitor?**
Hospital connection/training status, round progress, Dice/IoU/loss, privacy budget
(epsilon/delta/cumulative epsilon/budget status), CKKS/TLS status, and an event log —
never raw data, model weights, or any secret, structurally enforced by an explicit
payload allowlist checked at event-construction time
(`server/dashboard/events.py::ALLOWED_PAYLOAD_KEYS`).

**22. What are the major limitations?**
Simulated hospitals on one local machine (not a real multi-institution deployment); a
small, development-scale dataset (9 studies — not enough for clinical meaning); an
honest-but-curious threat model only (no defense against a malicious hospital or a
compromised server); no dashboard authentication; real, measurable CKKS/TLS overhead; a
real DP utility cost; and an unresolved `flwr run` live-deployment submission issue on
this specific Windows/Python 3.14 environment. Full list: README Section 11,
`docs/security.md`.

**23. What would you improve for production?**
A real multi-institution deployment (not simulated hospitals on one machine); the full
BraTS dataset (or comparable scale) instead of a 9-study development subset; a tighter
DP composition (RDP/moments accountant instead of basic composition); authentication and
TLS for the dashboard itself; resolving (or working around, e.g. on a Linux deployment
target) the `flwr run` live-submission issue; and an independent cryptography/security
audit before this system would be appropriate for any real patient data.
