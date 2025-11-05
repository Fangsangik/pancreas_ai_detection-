"""
OAR (Organs at Risk) Segmentation Model

Automatically segments critical organs near pancreas:
- Duodenum (가장 중요 - GI toxicity 주원인)
- Stomach
- Liver
- Small intestine
- Kidneys (left/right)
- Spinal cord

Architecture: nnU-Net inspired 3D U-Net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from .base import BaseRadiotherapyModel


class nnUNetBlock(nn.Module):
    """nnU-Net style residual block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return out


class OARSegmentationNetwork(nn.Module):
    """nnU-Net style 3D U-Net for OAR segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 7,  # Background + 6 OARs
        base_channels: int = 32
    ):
        super().__init__()

        # Encoder
        self.enc1 = nnUNetBlock(in_channels, base_channels)
        self.pool1 = nn.Conv3d(base_channels, base_channels, kernel_size=2, stride=2)

        self.enc2 = nnUNetBlock(base_channels, base_channels * 2)
        self.pool2 = nn.Conv3d(base_channels * 2, base_channels * 2,
                               kernel_size=2, stride=2)

        self.enc3 = nnUNetBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.Conv3d(base_channels * 4, base_channels * 4,
                               kernel_size=2, stride=2)

        self.enc4 = nnUNetBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.Conv3d(base_channels * 8, base_channels * 8,
                               kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nnUNetBlock(base_channels * 8, base_channels * 16)

        # Decoder
        self.upconv4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8,
                                          kernel_size=2, stride=2)
        self.dec4 = nnUNetBlock(base_channels * 16, base_channels * 8)

        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4,
                                          kernel_size=2, stride=2)
        self.dec3 = nnUNetBlock(base_channels * 8, base_channels * 4)

        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2,
                                          kernel_size=2, stride=2)
        self.dec2 = nnUNetBlock(base_channels * 4, base_channels * 2)

        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels,
                                          kernel_size=2, stride=2)
        self.dec1 = nnUNetBlock(base_channels * 2, base_channels)

        # Deep supervision outputs (optional)
        self.out_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)
        self.out_conv2 = nn.Conv3d(base_channels * 2, num_classes, kernel_size=1)
        self.out_conv3 = nn.Conv3d(base_channels * 4, num_classes, kernel_size=1)

    def forward(self, x, deep_supervision: bool = False):
        """
        Args:
            x: (B, 1, D, H, W) - CT scan
            deep_supervision: Return intermediate outputs

        Returns:
            segmentation: (B, num_classes, D, H, W)
        """
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)

        enc2 = self.enc2(x)
        x = self.pool2(enc2)

        enc3 = self.enc3(x)
        x = self.pool3(enc3)

        enc4 = self.enc4(x)
        x = self.pool4(enc4)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)

        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        dec3 = self.dec3(x)

        x = self.upconv2(dec3)
        x = torch.cat([x, enc2], dim=1)
        dec2 = self.dec2(x)

        x = self.upconv1(dec2)
        x = torch.cat([x, enc1], dim=1)
        dec1 = self.dec1(x)

        # Output
        out = self.out_conv(dec1)

        if deep_supervision:
            # Intermediate outputs for deep supervision
            out2 = self.out_conv2(dec2)
            out3 = self.out_conv3(dec3)

            # Upsample to match output size
            out2 = F.interpolate(out2, size=out.shape[2:], mode='trilinear',
                                align_corners=False)
            out3 = F.interpolate(out3, size=out.shape[2:], mode='trilinear',
                                align_corners=False)

            return out, out2, out3
        else:
            return out


class OARSegmentationModel(BaseRadiotherapyModel):
    """
    OAR (Organs at Risk) Segmentation Model.

    Segments critical organs for radiotherapy planning:
    - Class 0: Background
    - Class 1: Duodenum (십이지장) - most critical for toxicity
    - Class 2: Stomach (위)
    - Class 3: Small intestine (소장)
    - Class 4: Liver (간)
    - Class 5: Left kidney (왼쪽 신장)
    - Class 6: Right kidney (오른쪽 신장)

    Input: CT scan
    Output: Multi-class segmentation mask
    """

    OAR_NAMES = [
        'background',
        'duodenum',
        'stomach',
        'small_intestine',
        'liver',
        'left_kidney',
        'right_kidney'
    ]

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 7,
        base_channels: int = 32,
        deep_supervision: bool = True,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        **kwargs
    ):
        super().__init__(learning_rate, weight_decay, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        # Segmentation network
        self.network = OARSegmentationNetwork(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels
        )

    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 1, D, H, W) - CT scan

        Returns:
            segmentation logits and probabilities
        """
        if self.deep_supervision and self.training:
            out, out2, out3 = self.network(x, deep_supervision=True)

            return {
                'seg_logits': out,
                'seg_logits_2': out2,
                'seg_logits_3': out3,
                'seg_probs': F.softmax(out, dim=1)
            }
        else:
            out = self.network(x, deep_supervision=False)

            return {
                'seg_logits': out,
                'seg_probs': F.softmax(out, dim=1)
            }

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute segmentation loss.

        Loss = Dice loss + CE loss (+ deep supervision)
        """
        seg_logits = predictions['seg_logits']
        seg_target = targets['oar_mask'].long()

        # 1. Dice loss
        dice_loss = self._dice_loss(seg_logits, seg_target)

        # 2. Cross-entropy loss
        ce_loss = F.cross_entropy(seg_logits, seg_target)

        # Combined loss
        seg_loss = 0.5 * dice_loss + 0.5 * ce_loss

        losses = {
            'dice_loss': dice_loss,
            'ce_loss': ce_loss,
            'seg_loss': seg_loss
        }

        # Deep supervision
        if self.deep_supervision and self.training:
            if 'seg_logits_2' in predictions:
                dice_loss_2 = self._dice_loss(predictions['seg_logits_2'], seg_target)
                ce_loss_2 = F.cross_entropy(predictions['seg_logits_2'], seg_target)
                losses['seg_loss'] += 0.5 * (0.5 * dice_loss_2 + 0.5 * ce_loss_2)

            if 'seg_logits_3' in predictions:
                dice_loss_3 = self._dice_loss(predictions['seg_logits_3'], seg_target)
                ce_loss_3 = F.cross_entropy(predictions['seg_logits_3'], seg_target)
                losses['seg_loss'] += 0.25 * (0.5 * dice_loss_3 + 0.5 * ce_loss_3)

        losses['total_loss'] = losses['seg_loss']

        return losses

    def _dice_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1.0
    ) -> torch.Tensor:
        """
        Compute Dice loss for multi-class segmentation.

        Args:
            logits: (B, C, D, H, W)
            targets: (B, D, H, W)
            smooth: Smoothing factor

        Returns:
            Dice loss
        """
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()

        # Compute Dice per class
        dice_scores = []
        for c in range(self.num_classes):
            pred_c = probs[:, c]
            target_c = targets_one_hot[:, c]

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice)

        # Average Dice across classes (skip background)
        dice_loss = 1.0 - torch.stack(dice_scores[1:]).mean()

        return dice_loss

    def compute_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute segmentation metrics (Dice per organ).
        """
        seg_probs = predictions['seg_probs']
        seg_pred = torch.argmax(seg_probs, dim=1)
        seg_target = targets['oar_mask'].long()

        metrics = {}

        # Dice per organ
        for c in range(1, self.num_classes):  # Skip background
            pred_c = (seg_pred == c).float()
            target_c = (seg_target == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            if union > 0:
                dice = (2.0 * intersection / union).item()
            else:
                dice = 0.0

            organ_name = self.OAR_NAMES[c]
            metrics[f'dice_{organ_name}'] = dice

        # Average Dice
        dice_values = [v for k, v in metrics.items() if k.startswith('dice_')]
        metrics['dice_mean'] = sum(dice_values) / len(dice_values) if dice_values else 0.0

        return metrics
