"""Public-context fingerprinting: a real, structural way to reject a "wrong context"
update.

Discovered by testing (not assumed): CKKS decryption does not fail closed on a wrong
context. Linking a ciphertext to an independently-generated context and decrypting it
does not raise -- it silently returns meaningless numbers (observed: values on the order
of 1e31 for inputs of order 1.0). TenSEAL provides no ciphertext integrity/authentication
by default. So "wrong context is rejected" cannot be TenSEAL's job here -- this module
gives the aggregation server its own check: a SHA-256 fingerprint of the public context
bytes, computed once by whoever encrypts and verified by the aggregation server before it
ever aggregates an update, so a ciphertext encrypted under a different (unauthorized or
mismatched) context is caught structurally rather than silently corrupting the aggregate.
"""

from __future__ import annotations

import hashlib


def compute_context_fingerprint(public_context_bytes: bytes) -> str:
    return hashlib.sha256(public_context_bytes).hexdigest()
