"""
Segmentation Trainer
====================

PyTorch Lightning module for training segmentation models.
"""

import torch
import pytorch_lightning as pl
from typing import Dict
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric


class SegmentationTrainer(pl.LightningModule):
    """
    Lightning module for segmentation training.

    Handles training, validation, and metrics computation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        **kwargs
    ):
        """
        Args:
            model: Segmentation model
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Loss function
        self.loss_fn = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
        )

        # Metrics
        self.train_dice = DiceMetric(include_background=False, reduction="mean")
        self.val_dice = DiceMetric(include_background=False, reduction="mean")

    def forward(self, x):
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch: Dict, batch_idx: int):
        """Training step."""
        image = batch['image']
        label = batch['label']

        # Ensure 5D tensors: [B, C, D, H, W]
        if image.ndim == 4:  # [B, D, H, W]
            image = image.unsqueeze(1)  # [B, 1, D, H, W]
        if label.ndim == 4:  # [B, D, H, W]
            label = label.unsqueeze(1)  # [B, 1, D, H, W]

        # Forward pass
        pred = self.forward(image)

        # Compute loss
        loss = self.loss_fn(pred, label)

        # Compute metrics
        pred_binary = torch.argmax(pred, dim=1, keepdim=True)
        self.train_dice(pred_binary, label)

        # Log
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def on_training_epoch_end(self):
        """End of training epoch."""
        # Compute and log metrics
        dice = self.train_dice.aggregate()
        self.log('train_dice', dice)
        self.train_dice.reset()

    def validation_step(self, batch: Dict, batch_idx: int):
        """Validation step."""
        image = batch['image']
        label = batch['label']

        # Ensure 5D tensors: [B, C, D, H, W]
        if image.ndim == 4:  # [B, D, H, W]
            image = image.unsqueeze(1)  # [B, 1, D, H, W]
        if label.ndim == 4:  # [B, D, H, W]
            label = label.unsqueeze(1)  # [B, 1, D, H, W]

        # Forward pass
        pred = self.forward(image)

        # Compute loss
        loss = self.loss_fn(pred, label)

        # Compute metrics
        pred_binary = torch.argmax(pred, dim=1, keepdim=True)
        self.val_dice(pred_binary, label)

        # Log
        self.log('val_loss', loss, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        """End of validation epoch."""
        # Compute and log metrics
        dice = self.val_dice.aggregate()
        self.log('val_dice', dice, prog_bar=True)
        self.val_dice.reset()

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
