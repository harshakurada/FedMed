"""Centralized 3D U-Net training baseline: model, training/validation loops,
checkpointing, and inference -- built on the Module 3 `cv_model.brats` dataset
pipeline.

No Flower, TenSEAL, gRPC, or dashboard code lives here (or is imported here),
by design: this training logic must later be reusable, unmodified, as the
local-training step inside each federated hospital node.
"""
