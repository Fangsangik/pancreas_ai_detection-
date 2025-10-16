"""
Attention U-Net 3D Model
=========================

U-Net with attention gates for improved focus on relevant features.
Model 4 of 5 segmentation CNNs.
"""

import torch
import torch.nn as nn
from .base import BaseSegmentationModel


class AttentionUNet3D(BaseSegmentationModel):
    """
    3D Attention U-Net for pancreas segmentation.

    Key features:
    - Attention gates in skip connections
    - Focuses on relevant features
    - Suppresses irrelevant regions
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

        # TODO: Implement encoder blocks
        self.encoder = nn.ModuleList()

        # TODO: Implement attention gates
        self.attention_gates = nn.ModuleList()

        # TODO: Implement decoder blocks
        self.decoder = nn.ModuleList()

        # Final classification
        self.final_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention mechanism.

        Args:
            x: Input tensor [B, 1, D, H, W]

        Returns:
            Segmentation mask [B, num_classes, D, H, W]
        """
        # TODO: Implement forward pass with attention gates
        # 1. Encoder
        # 2. Attention-gated skip connections
        # 3. Decoder
        raise NotImplementedError("AttentionUNet3D forward pass not implemented")

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "base_channels": self.base_channels,
            "depth": self.depth,
        })
        return config
