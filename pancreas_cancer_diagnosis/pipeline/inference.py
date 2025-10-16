"""
End-to-End Inference Script
============================

Run complete pipeline: CT → 5 Seg Models → Ensemble Classification → Diagnosis

Usage:
    python -m pancreas_cancer_diagnosis.pipeline.inference \\
        --config configs/pipeline_inference.yaml \\
        --input data/test_ct.nii.gz \\
        --output results/prediction.json
"""

import argparse
import yaml
import torch
import nibabel as nib
from pathlib import Path
import json
import numpy as np

from .orchestrator import EndToEndPipeline
from ..segmentation.models import (
    UNet3D, ResUNet3D, VNet, AttentionUNet3D, C2FNAS
)
from ..classification.models import ResNet3D, DenseNet3D


SEG_MODEL_REGISTRY = {
    'unet': UNet3D,
    'resunet': ResUNet3D,
    'vnet': VNet,
    'attention_unet': AttentionUNet3D,
    'c2fnas': C2FNAS,
}

CLS_MODEL_REGISTRY = {
    'resnet3d': ResNet3D,
    'densenet3d': DenseNet3D,
}


def load_ct_image(path: str) -> torch.Tensor:
    """Load CT image from NIfTI file."""
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.float32)

    # Add batch and channel dimensions [B, C, D, H, W]
    data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)

    return data


def save_segmentation(seg_mask: np.ndarray, output_path: str, reference_nii):
    """Save segmentation mask as NIfTI file."""
    seg_nii = nib.Nifti1Image(seg_mask, reference_nii.affine, reference_nii.header)
    nib.save(seg_nii, output_path)
    print(f"Saved segmentation to {output_path}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="End-to-end pancreas cancer diagnosis"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to pipeline configuration file'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CT scan (.nii.gz)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--save_segmentations',
        action='store_true',
        help='Save segmentation outputs'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on'
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 80)
    print(f"Pancreas Cancer Diagnosis - End-to-End Pipeline")
    print(f"=" * 80)
    print(f"Input CT: {args.input}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"=" * 80)

    # Load 5 segmentation models
    print("\nLoading segmentation models...")
    seg_models = []
    for i, seg_config in enumerate(config['segmentation_models']):
        model_type = seg_config['type']
        checkpoint = seg_config['checkpoint']

        model_class = SEG_MODEL_REGISTRY[model_type]
        model = model_class(**seg_config.get('params', {}))
        model.load_checkpoint(checkpoint)

        seg_models.append(model)
        print(f"  [{i+1}/5] Loaded {model_type} from {checkpoint}")

    # Load 5 classification models
    print("\nLoading classification models...")
    cls_models = []
    for i, cls_config in enumerate(config['classification_models']):
        model_type = cls_config['type']
        checkpoint = cls_config['checkpoint']

        model_class = CLS_MODEL_REGISTRY[model_type]
        model = model_class(**cls_config.get('params', {}))
        model.load_checkpoint(checkpoint)

        cls_models.append(model)
        print(f"  [{i+1}/5] Loaded {model_type} from {checkpoint}")

    # Initialize pipeline
    print("\nInitializing end-to-end pipeline...")
    pipeline = EndToEndPipeline(
        seg_models=seg_models,
        cls_models=cls_models,
        ensemble_method=config.get('ensemble_method', 'average'),
        ensemble_weights=config.get('ensemble_weights'),
        device=args.device,
    )

    # Load CT image
    print(f"\nLoading CT image from {args.input}...")
    ct_image = load_ct_image(args.input)
    reference_nii = nib.load(args.input)

    # Run inference
    print("\n" + "=" * 80)
    print("Running end-to-end inference...")
    print("=" * 80)

    results = pipeline.predict(
        ct_image,
        return_segmentations=args.save_segmentations,
        return_probabilities=True,
        return_uncertainty=True,
    )

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    prediction = results['prediction'][0]
    class_name = results['class_names'][prediction]
    probs = results['probabilities'][0]

    print(f"\nDiagnosis: {class_name}")
    print(f"Confidence: {probs[prediction]*100:.2f}%")
    print(f"\nClass Probabilities:")
    for i, (name, prob) in enumerate(zip(results['class_names'], probs)):
        print(f"  {name}: {prob*100:.2f}%")

    if 'uncertainty' in results:
        uncertainty = results['uncertainty'][0]
        print(f"\nUncertainty (std across models):")
        for i, (name, unc) in enumerate(zip(results['class_names'], uncertainty)):
            print(f"  {name}: {unc:.4f}")

    # Save results
    output_file = output_dir / "prediction.json"
    results_dict = {
        'input_file': str(args.input),
        'prediction': int(prediction),
        'diagnosis': class_name,
        'probabilities': {
            name: float(prob)
            for name, prob in zip(results['class_names'], probs)
        },
    }

    if 'uncertainty' in results:
        results_dict['uncertainty'] = {
            name: float(unc)
            for name, unc in zip(results['class_names'], uncertainty)
        }

    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Save segmentations if requested
    if args.save_segmentations and 'segmentations' in results:
        print("\nSaving segmentation outputs...")
        seg_dir = output_dir / "segmentations"
        seg_dir.mkdir(exist_ok=True)

        for i, seg_mask in enumerate(results['segmentations']):
            seg_path = seg_dir / f"segmentation_model_{i+1}.nii.gz"
            save_segmentation(seg_mask[0, 0], str(seg_path), reference_nii)

    print("\n" + "=" * 80)
    print("Inference complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
