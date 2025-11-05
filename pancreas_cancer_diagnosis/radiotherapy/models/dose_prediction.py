"""
Dose Prediction Model for Radiotherapy Planning

Predicts optimal 3D dose distribution given:
- CT scan
- Tumor segmentation
- OAR segmentation
- Prescription dose

Architecture: 3D U-Net variant with attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .base import BaseRadiotherapyModel


class AttentionBlock3D(nn.Module):
    """Attention gate for U-Net."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: gating signal (from decoder)
            x: skip connection (from encoder)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class DoseUNet3D(nn.Module):
    """3D U-Net with attention for dose prediction."""

    def __init__(
        self,
        in_channels: int = 4,  # CT + tumor + OARs
        out_channels: int = 1,  # Dose map
        base_channels: int = 32
    ):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 16)

        # Decoder with attention
        self.upconv4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8,
                                          kernel_size=2, stride=2)
        self.att4 = AttentionBlock3D(base_channels * 8, base_channels * 8,
                                      base_channels * 4)
        self.dec4 = self._conv_block(base_channels * 16, base_channels * 8)

        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4,
                                          kernel_size=2, stride=2)
        self.att3 = AttentionBlock3D(base_channels * 4, base_channels * 4,
                                      base_channels * 2)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)

        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2,
                                          kernel_size=2, stride=2)
        self.att2 = AttentionBlock3D(base_channels * 2, base_channels * 2,
                                      base_channels)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)

        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels,
                                          kernel_size=2, stride=2)
        self.att1 = AttentionBlock3D(base_channels, base_channels,
                                      base_channels // 2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)

        # Output
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def _conv_block(self, in_channels: int, out_channels: int):
        """Double convolution block."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W) - CT + tumor + OARs

        Returns:
            dose: (B, 1, D, H, W) - predicted dose distribution
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

        # Decoder with attention
        x = self.upconv4(x)
        enc4_att = self.att4(x, enc4)
        x = torch.cat([x, enc4_att], dim=1)
        x = self.dec4(x)

        x = self.upconv3(x)
        enc3_att = self.att3(x, enc3)
        x = torch.cat([x, enc3_att], dim=1)
        x = self.dec3(x)

        x = self.upconv2(x)
        enc2_att = self.att2(x, enc2)
        x = torch.cat([x, enc2_att], dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        enc1_att = self.att1(x, enc1)
        x = torch.cat([x, enc1_att], dim=1)
        x = self.dec1(x)

        # Output
        dose = self.out_conv(x)

        return dose


class DosePredictionModel(BaseRadiotherapyModel):
    """
    Dose distribution prediction model.

    Predicts 3D dose distribution for radiotherapy planning.

    Input:
    - CT scan (1 channel)
    - Tumor segmentation (1 channel)
    - OAR segmentations (N channels) - e.g., duodenum, stomach, liver
    - Prescription dose (scalar)

    Output:
    - 3D dose distribution (Gy)
    - DVH statistics
    """

    def __init__(
        self,
        in_channels: int = 4,  # CT + tumor + 2 OARs
        base_channels: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        **kwargs
    ):
        super().__init__(learning_rate, weight_decay, **kwargs)

        self.in_channels = in_channels

        # Dose prediction network
        self.unet = DoseUNet3D(
            in_channels=in_channels,
            out_channels=1,
            base_channels=base_channels
        )

    def forward(
        self,
        x: torch.Tensor,
        prescription_dose: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, D, H, W) - CT + tumor + OARs
            prescription_dose: (B,) - target dose in Gy

        Returns:
            dose_map: (B, 1, D, H, W) - predicted dose
        """
        # Predict dose distribution
        dose_map = self.unet(x)

        # Ensure non-negative dose
        dose_map = F.relu(dose_map)

        # Normalize by prescription dose if provided
        if prescription_dose is not None:
            # Scale dose map
            max_dose = dose_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].max(dim=-3, keepdim=True)[0]
            dose_map = dose_map / (max_dose + 1e-8) * prescription_dose.view(-1, 1, 1, 1, 1)

        return {
            'dose_map': dose_map
        }

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute dose prediction loss.

        Loss components:
        1. MSE loss (voxel-wise)
        2. DVH loss (dose-volume constraints)
        3. Gradient loss (smoothness)
        """
        dose_pred = predictions['dose_map']
        dose_target = targets['dose_map']

        # 1. MSE loss
        mse_loss = F.mse_loss(dose_pred, dose_target)

        # 2. Gradient loss (encourage smooth dose distribution)
        grad_loss = self._gradient_loss(dose_pred)

        # 3. DVH loss (if OAR masks provided)
        dvh_loss = torch.tensor(0.0, device=dose_pred.device)
        if 'oar_masks' in targets:
            dvh_loss = self._dvh_loss(dose_pred, targets['oar_masks'],
                                      targets.get('oar_constraints'))

        # Total loss
        total_loss = mse_loss + 0.1 * grad_loss + 0.5 * dvh_loss

        return {
            'mse_loss': mse_loss,
            'grad_loss': grad_loss,
            'dvh_loss': dvh_loss,
            'total_loss': total_loss
        }

    def _gradient_loss(self, dose_map: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient loss for dose smoothness.

        Args:
            dose_map: (B, 1, D, H, W)

        Returns:
            gradient loss
        """
        # Compute gradients in 3D
        grad_d = torch.abs(dose_map[:, :, 1:, :, :] - dose_map[:, :, :-1, :, :])
        grad_h = torch.abs(dose_map[:, :, :, 1:, :] - dose_map[:, :, :, :-1, :])
        grad_w = torch.abs(dose_map[:, :, :, :, 1:] - dose_map[:, :, :, :, :-1])

        # Average gradient magnitude
        grad_loss = (grad_d.mean() + grad_h.mean() + grad_w.mean()) / 3.0

        return grad_loss

    def _dvh_loss(
        self,
        dose_map: torch.Tensor,
        oar_masks: torch.Tensor,
        constraints: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Compute DVH-based loss for OAR constraints.

        Args:
            dose_map: (B, 1, D, H, W)
            oar_masks: (B, N_oars, D, H, W)
            constraints: Dictionary of dose constraints

        Returns:
            DVH loss
        """
        if constraints is None:
            # Default: penalize high dose to OARs
            oar_dose = dose_map * oar_masks
            return oar_dose.mean()

        # Compute specific DVH constraints
        # e.g., D_mean < 30 Gy for duodenum
        dvh_loss = torch.tensor(0.0, device=dose_map.device)

        for oar_idx, (oar_name, constraint) in enumerate(constraints.items()):
            if oar_idx >= oar_masks.shape[1]:
                break

            oar_mask = oar_masks[:, oar_idx:oar_idx+1]
            oar_dose = dose_map * oar_mask

            # Mean dose constraint
            if 'mean' in constraint:
                mean_dose = oar_dose.sum(dim=[2, 3, 4]) / (oar_mask.sum(dim=[2, 3, 4]) + 1e-8)
                dvh_loss += F.relu(mean_dose - constraint['mean']).mean()

            # Max dose constraint
            if 'max' in constraint:
                max_dose = oar_dose.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                dvh_loss += F.relu(max_dose - constraint['max']).mean()

        return dvh_loss

    def compute_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute dose prediction metrics.
        """
        dose_pred = predictions['dose_map']
        dose_target = targets['dose_map']

        # MAE
        mae = torch.abs(dose_pred - dose_target).mean()

        # Max dose error
        max_pred = dose_pred.max()
        max_target = dose_target.max()
        max_error = torch.abs(max_pred - max_target)

        return {
            'dose_mae': mae.item(),
            'dose_max_error': max_error.item()
        }
