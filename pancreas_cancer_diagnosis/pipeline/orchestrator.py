"""
End-to-End Pipeline Orchestrator
=================================

Coordinates the full workflow:
1. Run 5 segmentation CNNs
2. Collect segmentation outputs
3. Run classification ensemble
4. Return final diagnosis
"""

import torch
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np

from ..segmentation.models import BaseSegmentationModel
from ..classification.models import BaseClassificationModel, EnsembleClassifier


class EndToEndPipeline:
    """
    Main pipeline orchestrator.

    Workflow:
    1. CT Image → 5 Segmentation Models → 5 Segmentation Masks
    2. 5 Segmentation Masks → Classification Ensemble → Final Diagnosis
    """

    def __init__(
        self,
        seg_models: List[BaseSegmentationModel],
        cls_models: List[BaseClassificationModel],
        ensemble_method: str = "average",
        ensemble_weights: Optional[List[float]] = None,
        device: str = "cuda",
    ):
        """
        Args:
            seg_models: List of 5 segmentation models
            cls_models: List of classification models (one per segmentation)
            ensemble_method: Method for ensemble ('average', 'weighted', 'voting', 'stacking')
            ensemble_weights: Optional weights for ensemble
            device: Device to run on
        """
        assert len(seg_models) == 5, "Expected 5 segmentation models"
        assert len(cls_models) == 5, "Expected 5 classification models (one per segmentation)"

        self.seg_models = seg_models
        self.cls_models = cls_models
        self.device = device

        # Move models to device
        for model in self.seg_models:
            model.to(device)
            model.eval()

        for model in self.cls_models:
            model.to(device)
            model.eval()

        # Create ensemble classifier
        self.ensemble = EnsembleClassifier(
            models=cls_models,
            ensemble_method=ensemble_method,
            weights=ensemble_weights,
            num_classes=2,
        ).to(device)
        self.ensemble.eval()

    @torch.no_grad()
    def predict(
        self,
        ct_image: torch.Tensor,
        return_segmentations: bool = False,
        return_probabilities: bool = False,
        return_uncertainty: bool = False,
    ) -> Dict:
        """
        Full end-to-end prediction.

        Args:
            ct_image: Input CT image [B, C, D, H, W]
            return_segmentations: Return segmentation outputs
            return_probabilities: Return classification probabilities
            return_uncertainty: Return uncertainty estimation

        Returns:
            Dictionary with predictions and optional outputs
        """
        # Ensure input is on correct device
        ct_image = ct_image.to(self.device)

        # Step 1: Run 5 segmentation models
        print("Running 5 segmentation models...")
        seg_outputs = []
        for i, seg_model in enumerate(self.seg_models):
            print(f"  Segmentation model {i+1}/5: {seg_model.model_name}")
            seg_out = seg_model(ct_image)
            seg_outputs.append(seg_out)

        # Step 2: Run classification on each segmentation output
        print("Running classification ensemble...")
        cls_predictions = []
        for i, (cls_model, seg_out) in enumerate(zip(self.cls_models, seg_outputs)):
            print(f"  Classification model {i+1}/5")
            pred = cls_model(seg_out)
            cls_predictions.append(pred)

        cls_predictions = torch.stack(cls_predictions, dim=0)

        # Step 3: Ensemble predictions
        if return_uncertainty:
            ensemble_result = self.ensemble.predict_proba_with_uncertainty(ct_image)
            final_probs = ensemble_result['mean']
            uncertainty = ensemble_result['std']
        else:
            # Use ensemble forward pass
            final_logits = self.ensemble._average_ensemble(cls_predictions)
            final_probs = torch.softmax(final_logits, dim=1)
            uncertainty = None

        final_class = torch.argmax(final_probs, dim=1)

        # Prepare results
        results = {
            'prediction': final_class.cpu().numpy(),
            'class_names': ['Normal', 'Cancer'],
        }

        if return_probabilities:
            results['probabilities'] = final_probs.cpu().numpy()

        if return_segmentations:
            results['segmentations'] = [seg.cpu().numpy() for seg in seg_outputs]

        if return_uncertainty and uncertainty is not None:
            results['uncertainty'] = uncertainty.cpu().numpy()

        return results

    def predict_batch(
        self,
        ct_images: List[torch.Tensor],
        **kwargs
    ) -> List[Dict]:
        """
        Batch prediction.

        Args:
            ct_images: List of CT images
            **kwargs: Arguments for predict()

        Returns:
            List of prediction dictionaries
        """
        results = []
        for ct_image in ct_images:
            result = self.predict(ct_image, **kwargs)
            results.append(result)
        return results

    def save_pipeline(self, save_dir: str):
        """
        Save entire pipeline (all models).

        Args:
            save_dir: Directory to save models
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save segmentation models
        seg_dir = save_dir / "segmentation"
        seg_dir.mkdir(exist_ok=True)
        for i, model in enumerate(self.seg_models):
            model_path = seg_dir / f"seg_model_{i}_{model.model_name}.pth"
            model.save_checkpoint(str(model_path))

        # Save classification models
        cls_dir = save_dir / "classification"
        cls_dir.mkdir(exist_ok=True)
        for i, model in enumerate(self.cls_models):
            model_path = cls_dir / f"cls_model_{i}_{model.model_name}.pth"
            model.save_checkpoint(str(model_path))

        print(f"Pipeline saved to {save_dir}")

    @classmethod
    def load_pipeline(
        cls,
        load_dir: str,
        seg_model_classes: List,
        cls_model_classes: List,
        device: str = "cuda",
    ):
        """
        Load entire pipeline from saved models.

        Args:
            load_dir: Directory with saved models
            seg_model_classes: List of 5 segmentation model classes
            cls_model_classes: List of 5 classification model classes
            device: Device to load on

        Returns:
            EndToEndPipeline instance
        """
        load_dir = Path(load_dir)

        # Load segmentation models
        seg_models = []
        seg_dir = load_dir / "segmentation"
        for i, model_class in enumerate(seg_model_classes):
            model = model_class()
            # Find checkpoint file
            checkpoint_files = list(seg_dir.glob(f"seg_model_{i}_*.pth"))
            if checkpoint_files:
                model.load_checkpoint(str(checkpoint_files[0]))
            seg_models.append(model)

        # Load classification models
        cls_models = []
        cls_dir = load_dir / "classification"
        for i, model_class in enumerate(cls_model_classes):
            model = model_class()
            checkpoint_files = list(cls_dir.glob(f"cls_model_{i}_*.pth"))
            if checkpoint_files:
                model.load_checkpoint(str(checkpoint_files[0]))
            cls_models.append(model)

        return cls(
            seg_models=seg_models,
            cls_models=cls_models,
            device=device,
        )


class ReproducibilityManager:
    """
    Manager for ensuring reproducibility across 5 model runs.

    Tracks configurations, seeds, and results for each model.
    """

    def __init__(self, experiment_dir: str):
        """
        Args:
            experiment_dir: Directory to save experiment info
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.seg_configs = []
        self.cls_configs = []
        self.results = {}

    def log_segmentation_config(self, model_idx: int, config: Dict):
        """Log segmentation model configuration."""
        self.seg_configs.append({
            'model_idx': model_idx,
            'config': config,
        })

    def log_classification_config(self, model_idx: int, config: Dict):
        """Log classification model configuration."""
        self.cls_configs.append({
            'model_idx': model_idx,
            'config': config,
        })

    def log_results(self, split: str, metrics: Dict):
        """Log evaluation results."""
        self.results[split] = metrics

    def save_experiment_info(self):
        """Save all experiment information."""
        import json

        # Save configs
        with open(self.experiment_dir / "seg_configs.json", 'w') as f:
            json.dump(self.seg_configs, f, indent=2)

        with open(self.experiment_dir / "cls_configs.json", 'w') as f:
            json.dump(self.cls_configs, f, indent=2)

        # Save results
        with open(self.experiment_dir / "results.json", 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Experiment info saved to {self.experiment_dir}")
