"""
ResNet3D Classification Model
==============================

3D ResNet for pancreatic cancer classification.
Can work with raw CT or segmentation masks.
"""

import torch
import torch.nn as nn
from .base import BaseClassificationModel


class ResNet3D(BaseClassificationModel):
    """
    3D ResNet for binary classification (normal vs cancer).

    Features:
    - Residual blocks for deep networks
    - Can accept raw CT or segmentation masks
    - Flexible input channels
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        layers: list = [2, 2, 2, 2],
        base_channels: int = 64,
        **kwargs
    ):
        """
        Args:
            in_channels: Input channels (1 for CT, 3 for seg masks)
            num_classes: Number of classes (2: normal/cancer)
            layers: Number of blocks in each stage
            base_channels: Base number of channels
        """
        super().__init__(in_channels, num_classes, **kwargs)

        self.layers = layers
        self.base_channels = base_channels

        # TODO: Implement ResNet3D layers
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # TODO: Implement residual layers
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, D, H, W]

        Returns:
            Logits [B, num_classes]
        """
        # TODO: Implement forward pass
        raise NotImplementedError("ResNet3D forward pass not implemented")

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "layers": self.layers,
            "base_channels": self.base_channels,
        })
        return config
