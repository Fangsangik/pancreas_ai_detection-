"""
Standalone Segmentation Training Script
========================================

Can be run independently to train any of the 5 segmentation models.

Usage:
    python -m pancreas_cancer_diagnosis.segmentation.training.train \\
        --config configs/seg_unet.yaml \\
        --model unet \\
        --gpus 4
"""

import argparse
import torch
import yaml
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from ..models import (
    UNet3D, ResUNet3D, VNet, AttentionUNet3D, C2FNAS
)
from ...data import SegmentationDataModule
from .trainer import SegmentationTrainer


MODEL_REGISTRY = {
    'unet': UNet3D,
    'resunet': ResUNet3D,
    'vnet': VNet,
    'attention_unet': AttentionUNet3D,
    'c2fnas': C2FNAS,
}


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train segmentation model independently"
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
        default='outputs/segmentation',
        help='Output directory for checkpoints and logs'
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
    print(f"Training Segmentation Model: {args.model}")
    print(f"Configuration: {args.config}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"=" * 80)

    # Initialize model
    model_class = MODEL_REGISTRY[args.model]
    model = model_class(**config['model'])

    # Initialize data module
    # CTDataModule에 없는 cache_rate 인자를 config에서 제거
    config['data'].pop('cache_rate', None)
    data_module = SegmentationDataModule(**config['data'])

    # Initialize trainer wrapper
    trainer_wrapper = SegmentationTrainer(
        model=model,
        **config['training']
    )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / 'checkpoints',
            filename='best_{epoch}_{val_dice:.4f}',
            monitor='val_dice',
            mode='max',
            save_top_k=3,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    # Initialize PyTorch Lightning trainer
    # 가속기 자동 선택 (CUDA, MPS, CPU 순)
    if args.gpus == 0:
        # Force CPU
        accelerator = 'cpu'
        devices = 1
    elif args.gpus > 0 and torch.cuda.is_available():
        accelerator = 'gpu'
        devices = args.gpus
    elif torch.backends.mps.is_available():
        accelerator = 'mps'
        devices = 1
    else:
        accelerator = 'cpu'
        devices = 1

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        accelerator=accelerator,
        devices=devices,
        strategy='auto',
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
