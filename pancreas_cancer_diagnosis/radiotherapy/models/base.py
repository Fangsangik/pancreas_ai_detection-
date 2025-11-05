"""
Base class for all radiotherapy models.

All radiotherapy models (multi-task, dose prediction, OAR segmentation)
should inherit from this base class for consistency.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class BaseRadiotherapyModel(pl.LightningModule, ABC):
    """
    Abstract base class for radiotherapy models.

    Provides common functionality:
    - Training/validation/test loops
    - Metric tracking
    - Checkpoint management
    - Logging
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, D, H, W)
            **kwargs: Additional inputs (e.g., clinical features)

        Returns:
            Dictionary of predictions
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for the model.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Dictionary of losses (must include 'total_loss')
        """
        pass

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Unpack batch
        images = batch['image']
        targets = {k: v for k, v in batch.items() if k != 'image'}

        # Forward pass
        predictions = self(images, **targets)

        # Compute loss
        losses = self.compute_loss(predictions, targets)

        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'train/{loss_name}', loss_value,
                    on_step=True, on_epoch=True, prog_bar=True)

        return losses['total_loss']

    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Unpack batch
        images = batch['image']
        targets = {k: v for k, v in batch.items() if k != 'image'}

        # Forward pass
        predictions = self(images, **targets)

        # Compute loss
        losses = self.compute_loss(predictions, targets)

        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(f'val/{loss_name}', loss_value,
                    on_step=False, on_epoch=True, prog_bar=True)

        # Compute and log metrics
        metrics = self.compute_metrics(predictions, targets)
        for metric_name, metric_value in metrics.items():
            self.log(f'val/{metric_name}', metric_value,
                    on_step=False, on_epoch=True)

        return losses['total_loss']

    def test_step(self, batch: Tuple, batch_idx: int) -> Dict[str, Any]:
        """Test step."""
        # Unpack batch
        images = batch['image']
        targets = {k: v for k, v in batch.items() if k != 'image'}

        # Forward pass
        predictions = self(images, **targets)

        # Compute loss
        losses = self.compute_loss(predictions, targets)

        # Compute metrics
        metrics = self.compute_metrics(predictions, targets)

        # Log everything
        for loss_name, loss_value in losses.items():
            self.log(f'test/{loss_name}', loss_value)

        for metric_name, metric_value in metrics.items():
            self.log(f'test/{metric_name}', metric_value)

        return {
            'predictions': predictions,
            'targets': targets,
            'metrics': metrics
        }

    def compute_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Override this in subclasses for task-specific metrics.

        Args:
            predictions: Model predictions
            targets: Ground truth targets

        Returns:
            Dictionary of metric values
        """
        return {}

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/total_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def predict_step(
        self,
        batch: Tuple,
        batch_idx: int,
        dataloader_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Prediction step for inference."""
        images = batch['image'] if isinstance(batch, dict) else batch
        predictions = self(images)
        return predictions
