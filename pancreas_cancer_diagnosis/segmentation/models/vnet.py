"""
V-Net Model
===========

V-Net architecture optimized for 3D medical image segmentation.
Model 3 of 5 segmentation CNNs.
"""

import torch
import torch.nn as nn
from .base import BaseSegmentationModel


class VNet(BaseSegmentationModel):
    """
    V-Net for pancreas segmentation.

    Key features:
    - Residual connections within each stage
    - PReLU activation
    - Optimized for volumetric segmentation
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 16,
        **kwargs
    ):
        """
        Args:
            in_channels: Input channels (1 for CT)
            num_classes: Output classes
            base_channels: Base number of channels
        """
        super().__init__(in_channels, num_classes, **kwargs)

        self.base_channels = base_channels

        # TODO: Implement V-Net stages
        # Typically: 5 stages with increasing channels
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Final classification
        self.final_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 1, D, H, W]

        Returns:
            Segmentation mask [B, num_classes, D, H, W]
        """
        # TODO: Implement V-Net forward pass
        raise NotImplementedError("VNet forward pass not implemented")

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "base_channels": self.base_channels,
        })
        return config
