"""Module 9: TenSEAL CKKS homomorphic encryption for federated model updates.

Entirely additive to Module 7's plaintext FedAvg path (server/federated/experiment.py,
untouched) -- this package only adds what happens after a hospital's local `fit()`
produces plaintext NDArrays: encrypt -> transport -> homomorphic aggregate -> decrypt ->
reconstruct. See docs/homomorphic_encryption.md for the full write-up (threat model, key
ownership, why the aggregation server can never decrypt an individual update).
"""
