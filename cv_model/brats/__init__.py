"""Local-filesystem BraTS pipeline: discovery, validation, transforms, PyTorch Dataset.

Separate from `cv_model/dataset.py` (which wraps MONAI's auto-downloading
`DecathlonDataset` for the MSD Task01_BrainTumour release). This subpackage
instead discovers a BraTS dataset the caller has already extracted locally
(e.g. a Kaggle BraTS2020 mirror) -- no network access, no auto-download.

Not imported by `cv_model/__init__.py` on purpose: importing this subpackage
pulls in `nibabel`, which callers that only need the Decathlon path (or the
model/params modules) don't need to load.
"""
