"""FedMed computer-vision module: 3D U-Net model + BraTS data pipeline (PyTorch/MONAI).

Re-exports the public API so callers can do `from cv_model import build_unet, ...`
instead of reaching into individual submodules.
"""

from cv_model.config import DEFAULT_CONFIG, BraTSConfig
from cv_model.dataset import get_dataloaders, partition_indices
from cv_model.model import build_dice_metric, build_loss_function, build_unet, count_parameters

__all__ = [
    "BraTSConfig",
    "DEFAULT_CONFIG",
    "get_dataloaders",
    "partition_indices",
    "build_unet",
    "build_loss_function",
    "build_dice_metric",
    "count_parameters",
]
