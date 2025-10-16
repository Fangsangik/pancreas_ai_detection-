"""
Classification Trainer
======================

PyTorch Lightning module for training classification models.
"""

import torch
import pytorch_lightning as pl
from typing import Dict
from torchmetrics import Accuracy, AUROC, F1Score


class ClassificationTrainer(pl.LightningModule):
    """
    Lightning module for classification training.

    Handles training, validation, and metrics computation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        num_classes: int = 2,
        **kwargs
    ):
        """
        Args:
            model: Classification model
            learning_rate: Learning rate
            weight_decay: Weight decay
            num_classes: Number of classes
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        # Loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
        self.val_auroc = AUROC(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)

    def forward(self, x):
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch: Dict, batch_idx: int):
        """Training step."""
        image = batch['image']
        label = batch['class_label']

        # Forward pass
        logits = self.forward(image)

        # Compute loss
        loss = self.loss_fn(logits, label)

        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, label)

        # Log
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        """Validation step."""
        image = batch['image']
        label = batch['class_label']

        # Forward pass
        logits = self.forward(image)

        # Compute loss
        loss = self.loss_fn(logits, label)

        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        self.val_acc(preds, label)
        self.val_auroc(probs[:, 1] if self.num_classes == 2 else probs, label)
        self.val_f1(preds, label)

        # Log
        self.log('val_loss', loss, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        """End of validation epoch."""
        # Log metrics
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_auroc', self.val_auroc.compute())
        self.log('val_f1', self.val_f1.compute())

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
