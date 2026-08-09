"""Module 8: a minimal, genuinely-mutual-TLS gRPC coordination service.

Not the transport for FedAvg's model-parameter/metric traffic -- that stays Flower's own
gRPC (its SuperLink/SuperNode deployment engine), configured with real one-way TLS via
its native `--ssl-*`/`--root-certificates` flags. See docs/secure_communication.md for
why the two are separate and neither duplicates the other.
"""
