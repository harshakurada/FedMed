"""Module 10: client-level (hospital-level) Differential Privacy for federated model
updates.

**Privacy unit, stated precisely:** this protects what one hospital's aggregate local
update reveals about that hospital's participation -- NOT individual patients, NOT
individual training examples/slices. See docs/differential_privacy.md for the full
explanation of why (a hospital's local training touches many patients and many patches
per patient) and for the complete threat model.

Entirely additive to Module 9 (server/federated/encrypted/, untouched) and Module 7
(server/federated/experiment.py, untouched): this package only changes *which numbers*
get handed to Module 9's encrypt_model_update -- clip -> noise -> reconstruct, applied at
the hospital, before encryption.
"""
