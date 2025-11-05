"""
Multi-Task Learning Model for Radiotherapy Outcome Prediction

Simultaneously predicts:
1. Survival time (regression)
2. Toxicity grade (multi-class classification)
3. Treatment response (binary classification)

Architecture:
- Shared 3D CNN encoder (ResNet3D or DenseNet3D)
- Task-specific heads
- Uncertainty estimation per task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from .base import BaseRadiotherapyModel


class SharedEncoder3D(nn.Module):
    """
    Shared 3D CNN encoder for multi-task learning.

    Extracts features from CT + tumor mask + OAR masks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_blocks: int = 4
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels

        # Initial convolution
        self.conv_in = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        # Residual blocks
        self.blocks = nn.ModuleList()
        channels = base_channels
        for i in range(num_blocks):
            out_channels = channels * 2
            self.blocks.append(
                ResidualBlock3D(channels, out_channels, stride=2 if i > 0 else 1)
            )
            channels = out_channels

        self.out_channels = channels

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool3d(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, D, H, W) - CT + masks

        Returns:
            features: (B, out_channels) - global features
            feature_map: (B, out_channels, D', H', W') - spatial features
        """
        # Initial conv
        x = self.conv_in(x)  # (B, base_channels, D/4, H/4, W/4)

        # Residual blocks
        for block in self.blocks:
            x = block(x)  # Progressively downsample

        # Global features
        features = self.gap(x).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, out_channels)

        return features, x


class ResidualBlock3D(nn.Module):
    """3D Residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)

        return out


class SurvivalHead(nn.Module):
    """
    Survival prediction head.

    Predicts:
    - Survival time (regression)
    - Uncertainty (aleatoric + epistemic)
    """

    def __init__(self, in_features: int, clinical_features: int = 10):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features + clinical_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Mean prediction
        self.fc_mean = nn.Linear(128, 1)

        # Log variance (for uncertainty)
        self.fc_log_var = nn.Linear(128, 1)

    def forward(
        self,
        features: torch.Tensor,
        clinical: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, in_features)
            clinical: (B, clinical_features) - age, stage, etc.

        Returns:
            survival_time: (B, 1) - predicted survival in months
            log_var: (B, 1) - log variance for uncertainty
        """
        # Concatenate imaging and clinical features
        if clinical is not None:
            x = torch.cat([features, clinical], dim=1)
        else:
            # Use zeros if no clinical features
            x = torch.cat([features, torch.zeros(features.size(0), 10,
                          device=features.device)], dim=1)

        x = self.fc(x)

        survival_time = self.fc_mean(x)  # (B, 1)
        log_var = self.fc_log_var(x)     # (B, 1)

        return {
            'survival_time': survival_time,
            'survival_uncertainty': log_var
        }


class ToxicityHead(nn.Module):
    """
    Toxicity prediction head.

    Predicts toxicity grade: 0 (none), 1 (mild), 2 (moderate), 3+ (severe)
    """

    def __init__(self, in_features: int, num_classes: int = 4):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, in_features)

        Returns:
            toxicity_logits: (B, num_classes)
            toxicity_probs: (B, num_classes)
        """
        logits = self.fc(features)  # (B, num_classes)
        probs = F.softmax(logits, dim=1)

        return {
            'toxicity_logits': logits,
            'toxicity_probs': probs
        }


class ResponseHead(nn.Module):
    """
    Treatment response prediction head.

    Binary classification: responder vs non-responder
    """

    def __init__(self, in_features: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, in_features)

        Returns:
            response_logit: (B, 1)
            response_prob: (B, 1)
        """
        logit = self.fc(features)  # (B, 1)
        prob = torch.sigmoid(logit)

        return {
            'response_logit': logit,
            'response_prob': prob
        }


class MultiTaskRadiotherapyModel(BaseRadiotherapyModel):
    """
    Multi-Task Learning model for radiotherapy outcome prediction.

    Predicts:
    1. Survival time (months)
    2. Toxicity grade (0-3+)
    3. Treatment response (binary)

    Input:
    - CT scan
    - Tumor segmentation mask
    - OAR segmentation masks (optional)
    - Clinical features (optional)

    Output:
    - Survival prediction + uncertainty
    - Toxicity grade probabilities
    - Response probability
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_blocks: int = 4,
        clinical_features: int = 10,
        num_toxicity_classes: int = 4,
        task_weights: Optional[Dict[str, float]] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        **kwargs
    ):
        super().__init__(learning_rate, weight_decay, **kwargs)

        self.in_channels = in_channels
        self.clinical_features = clinical_features
        self.num_toxicity_classes = num_toxicity_classes

        # Task weights (for loss balancing)
        if task_weights is None:
            task_weights = {
                'survival': 1.0,
                'toxicity': 1.0,
                'response': 1.0
            }
        self.task_weights = task_weights

        # Shared encoder
        self.encoder = SharedEncoder3D(
            in_channels=in_channels,
            base_channels=base_channels,
            num_blocks=num_blocks
        )

        # Task-specific heads
        encoder_out_channels = self.encoder.out_channels

        self.survival_head = SurvivalHead(
            in_features=encoder_out_channels,
            clinical_features=clinical_features
        )

        self.toxicity_head = ToxicityHead(
            in_features=encoder_out_channels,
            num_classes=num_toxicity_classes
        )

        self.response_head = ResponseHead(
            in_features=encoder_out_channels
        )

    def forward(
        self,
        x: torch.Tensor,
        clinical: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, D, H, W) - CT + masks
            clinical: (B, clinical_features) - clinical variables

        Returns:
            Dictionary with all predictions
        """
        # Shared encoding
        features, feature_map = self.encoder(x)

        # Task-specific predictions
        survival_pred = self.survival_head(features, clinical)
        toxicity_pred = self.toxicity_head(features)
        response_pred = self.response_head(features)

        # Combine all predictions
        predictions = {
            **survival_pred,
            **toxicity_pred,
            **response_pred,
            'features': features  # For analysis
        }

        return predictions

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Loss = w1 * L_survival + w2 * L_toxicity + w3 * L_response
        """
        losses = {}

        # 1. Survival loss (Gaussian NLL with uncertainty)
        if 'survival_time' in targets:
            survival_pred = predictions['survival_time']
            survival_target = targets['survival_time']
            log_var = predictions['survival_uncertainty']

            # Negative log likelihood
            survival_loss = 0.5 * (
                torch.exp(-log_var) * (survival_pred - survival_target) ** 2
                + log_var
            ).mean()

            losses['survival_loss'] = survival_loss

        # 2. Toxicity loss (Cross-entropy)
        if 'toxicity_grade' in targets:
            toxicity_logits = predictions['toxicity_logits']
            toxicity_target = targets['toxicity_grade'].long()

            toxicity_loss = F.cross_entropy(toxicity_logits, toxicity_target)
            losses['toxicity_loss'] = toxicity_loss

        # 3. Response loss (BCE)
        if 'response' in targets:
            response_logit = predictions['response_logit']
            response_target = targets['response'].float().unsqueeze(1)

            response_loss = F.binary_cross_entropy_with_logits(
                response_logit,
                response_target
            )
            losses['response_loss'] = response_loss

        # Total loss (weighted sum)
        total_loss = sum(
            self.task_weights.get(task.replace('_loss', ''), 1.0) * loss
            for task, loss in losses.items()
        )

        losses['total_loss'] = total_loss

        return losses

    def compute_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics for each task.
        """
        metrics = {}

        # Survival metrics (MAE, RMSE)
        if 'survival_time' in targets:
            survival_pred = predictions['survival_time']
            survival_target = targets['survival_time']

            mae = torch.abs(survival_pred - survival_target).mean()
            rmse = torch.sqrt(((survival_pred - survival_target) ** 2).mean())

            metrics['survival_mae'] = mae.item()
            metrics['survival_rmse'] = rmse.item()

        # Toxicity metrics (Accuracy)
        if 'toxicity_grade' in targets:
            toxicity_probs = predictions['toxicity_probs']
            toxicity_target = targets['toxicity_grade'].long()

            toxicity_pred = torch.argmax(toxicity_probs, dim=1)
            accuracy = (toxicity_pred == toxicity_target).float().mean()

            metrics['toxicity_accuracy'] = accuracy.item()

        # Response metrics (Accuracy, AUC would need full batch)
        if 'response' in targets:
            response_prob = predictions['response_prob']
            response_target = targets['response'].float()

            response_pred = (response_prob > 0.5).float().squeeze()
            accuracy = (response_pred == response_target).float().mean()

            metrics['response_accuracy'] = accuracy.item()

        return metrics
