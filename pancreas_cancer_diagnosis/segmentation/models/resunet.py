"""
Residual U-Net 3D Model
========================

U-Net with residual connections for improved gradient flow.
Model 2 of 5 segmentation CNNs.
"""

import torch
import torch.nn as nn
from .base import BaseSegmentationModel


class ResUNet3D(BaseSegmentationModel):
    """
    3D Residual U-Net for pancreas segmentation.

    Improvements over vanilla U-Net:
    - Residual blocks in encoder/decoder
    - Better gradient flow
    - Deeper networks possible
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 32,
        depth: int = 4,
        **kwargs
    ):
        """
        Args:
            in_channels: Input channels (1 for CT)
            num_classes: Output classes
            base_channels: Base number of channels
            depth: Number of encoder/decoder levels
        """
        super().__init__(in_channels, num_classes, **kwargs)

        self.base_channels = base_channels
        self.depth = depth

        # TODO: Implement residual encoder blocks
        self.encoder = nn.ModuleList()

        # TODO: Implement residual bottleneck
        self.bottleneck = None

        # TODO: Implement residual decoder blocks
        self.decoder = nn.ModuleList()

        # Final classification
        self.final_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: Input tensor [B, 1, D, H, W]

        Returns:
            Segmentation mask [B, num_classes, D, H, W]
        """
        # TODO: Implement forward pass with residual connections
        raise NotImplementedError("ResUNet3D forward pass not implemented")

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "base_channels": self.base_channels,
            "depth": self.depth,
        })
        return config
