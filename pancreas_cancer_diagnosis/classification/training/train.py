"""
Standalone Classification Training Script
==========================================

Can be run independently to train classification models.

Usage:
    python -m pancreas_cancer_diagnosis.classification.training.train \\
        --config configs/cls_resnet.yaml \\
        --model resnet3d \\
        --gpus 1
"""

import argparse
import yaml
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from ..models import ResNet3D, DenseNet3D, EnsembleClassifier
from ...data import ClassificationDataModule
from .trainer import ClassificationTrainer


MODEL_REGISTRY = {
    'resnet3d': ResNet3D,
    'densenet3d': DenseNet3D,
}


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train classification model independently"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help='Model architecture to use'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='Number of GPUs to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/classification',
        help='Output directory for checkpoints and logs'
    )
    parser.add_argument(
        '--use_segmentation',
        action='store_true',
        help='Use segmentation masks as input'
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed for reproducibility
    pl.seed_everything(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 80)
    print(f"Training Classification Model: {args.model}")
    print(f"Configuration: {args.config}")
    print(f"Output directory: {output_dir}")
    print(f"Use segmentation: {args.use_segmentation}")
    print(f"Random seed: {args.seed}")
    print(f"=" * 80)

    # Initialize model
    model_class = MODEL_REGISTRY[args.model]
    model = model_class(**config['model'])

    # Initialize data module
    data_module = ClassificationDataModule(
        use_segmentation=args.use_segmentation,
        **config['data']
    )

    # Initialize trainer wrapper
    trainer_wrapper = ClassificationTrainer(
        model=model,
        **config['training']
    )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / 'checkpoints',
            filename='best_{epoch}_{val_acc:.4f}',
            monitor='val_acc',
            mode='max',
            save_top_k=3,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    # Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus,
        max_epochs=config['training'].get('max_epochs', 100),
        callbacks=callbacks,
        log_every_n_steps=10,
        **config.get('trainer', {})
    )

    # Train model
    print("Starting training...")
    trainer.fit(trainer_wrapper, data_module)

    print(f"Training complete! Best model saved to {output_dir / 'checkpoints'}")


if __name__ == '__main__':
    main()
