"""Segmentation Models - 5 Independent CNN Architectures"""

from .base import BaseSegmentationModel
from .unet import UNet3D
from .resunet import ResUNet3D
from .vnet import VNet
from .attention_unet import AttentionUNet3D
from .c2fnas import C2FNAS

# Alias for backward compatibility
VNet3D = VNet

__all__ = [
    "BaseSegmentationModel",
    "UNet3D",
    "ResUNet3D",
    "VNet",
    "VNet3D",
    "AttentionUNet3D",
    "C2FNAS"
]
