"""
End-to-End Radiotherapy Pipeline
=================================

전체 워크플로우를 통합 실행:
1. OAR Segmentation (자동 장기 세그멘테이션)
2. Dose Prediction (선량 분포 예측)
3. Multi-task Prediction (생존/독성/반응 예측)
"""

import argparse
import json
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, \
    Orientationd, ScaleIntensityRanged, Resized, EnsureTyped

from .models import (
    OARSegmentationModel,
    DosePredictionModel,
    MultiTaskRadiotherapyModel
)


class RadiotherapyPipeline:
    """
    방사선 치료 계획 End-to-End 파이프라인.

    사용법:
    ```python
    pipeline = RadiotherapyPipeline(
        oar_checkpoint="path/to/oar.ckpt",
        dose_checkpoint="path/to/dose.ckpt",
        multitask_checkpoint="path/to/multitask.ckpt"
    )

    results = pipeline.run(
        ct_path="patient_ct.nii.gz",
        tumor_mask_path="tumor.nii.gz",
        clinical_data={"age": 65, "stage": 2, ...},
        prescription_dose=40.0
    )
    ```
    """

    def __init__(
        self,
        oar_checkpoint: str,
        dose_checkpoint: str,
        multitask_checkpoint: str,
        device: str = "cuda"
    ):
        """
        Args:
            oar_checkpoint: OAR segmentation 모델 체크포인트
            dose_checkpoint: Dose prediction 모델 체크포인트
            multitask_checkpoint: Multi-task 모델 체크포인트
            device: 디바이스 ("cuda" or "cpu")
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

        print(f"🚀 Radiotherapy Pipeline 초기화 중... (device: {self.device})")

        # 모델 로드
        print("  [1/3] OAR Segmentation 모델 로딩...")
        self.oar_model = OARSegmentationModel.load_from_checkpoint(
            oar_checkpoint,
            map_location=self.device
        )
        self.oar_model.to(self.device)
        self.oar_model.eval()

        print("  [2/3] Dose Prediction 모델 로딩...")
        self.dose_model = DosePredictionModel.load_from_checkpoint(
            dose_checkpoint,
            map_location=self.device
        )
        self.dose_model.to(self.device)
        self.dose_model.eval()

        print("  [3/3] Multi-task 모델 로딩...")
        self.multitask_model = MultiTaskRadiotherapyModel.load_from_checkpoint(
            multitask_checkpoint,
            map_location=self.device
        )
        self.multitask_model.to(self.device)
        self.multitask_model.eval()

        print("✅ 모든 모델 로드 완료!\n")

        # Transforms
        self._setup_transforms()

    def _setup_transforms(self):
        """전처리 파이프라인 설정"""
        # OAR segmentation transform
        self.oar_transform = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-200, a_max=200,
                b_min=0.0, b_max=1.0,
                clip=True
            ),
            Resized(keys=["image"], spatial_size=(128, 128, 128), mode="trilinear"),
            EnsureTyped(keys=["image"])
        ])

        # Multi-task transform
        self.multitask_transform = Compose([
            LoadImaged(keys=["image", "tumor_mask"]),
            EnsureChannelFirstd(keys=["image", "tumor_mask"]),
            Spacingd(
                keys=["image", "tumor_mask"],
                pixdim=(1.5, 1.5, 1.5),
                mode=("bilinear", "nearest")
            ),
            Orientationd(keys=["image", "tumor_mask"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-200, a_max=200,
                b_min=0.0, b_max=1.0,
                clip=True
            ),
            Resized(
                keys=["image", "tumor_mask"],
                spatial_size=(96, 96, 96),
                mode=("trilinear", "nearest")
            ),
            EnsureTyped(keys=["image", "tumor_mask"])
        ])

    def _segment_oars(self, ct_path: str) -> np.ndarray:
        """
        Step 1: OAR 세그멘테이션.

        Args:
            ct_path: CT scan 경로

        Returns:
            oar_mask: (D, H, W) numpy array (class indices 0-6)
        """
        print("  [Step 1/3] OAR Segmentation 실행 중...")

        data_dict = {"image": ct_path}
        data_dict = self.oar_transform(data_dict)

        image = data_dict["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.oar_model(image)

        seg_probs = predictions["seg_probs"]
        seg_pred = torch.argmax(seg_probs, dim=1)
        oar_mask = seg_pred.cpu().squeeze().numpy().astype(np.uint8)

        print(f"    ✅ OAR segmentation 완료 (unique classes: {np.unique(oar_mask)})")
        return oar_mask

    def _predict_dose(
        self,
        ct_path: str,
        tumor_mask_path: str,
        oar_mask: np.ndarray,
        prescription_dose: float
    ) -> np.ndarray:
        """
        Step 2: Dose distribution 예측.

        Args:
            ct_path: CT scan 경로
            tumor_mask_path: Tumor mask 경로
            oar_mask: OAR segmentation (D, H, W)
            prescription_dose: 처방 선량 (Gy)

        Returns:
            dose_map: (D, H, W) numpy array (Gy 단위)
        """
        print("  [Step 2/3] Dose Prediction 실행 중...")

        # OAR mask에서 duodenum(1), stomach(2) 추출
        duodenum_mask = (oar_mask == 1).astype(np.float32)
        stomach_mask = (oar_mask == 2).astype(np.float32)

        # Temporary NIfTI 파일로 저장 (transform이 파일 경로를 기대함)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f_duo:
            duodenum_path = f_duo.name
            nib.save(nib.Nifti1Image(duodenum_mask, np.eye(4)), duodenum_path)

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f_sto:
            stomach_path = f_sto.name
            nib.save(nib.Nifti1Image(stomach_mask, np.eye(4)), stomach_path)

        # Transform
        dose_transform = Compose([
            LoadImaged(keys=["image", "tumor_mask", "oar_mask_0", "oar_mask_1"]),
            EnsureChannelFirstd(keys=["image", "tumor_mask", "oar_mask_0", "oar_mask_1"]),
            Spacingd(
                keys=["image", "tumor_mask", "oar_mask_0", "oar_mask_1"],
                pixdim=(2.0, 2.0, 2.0),
                mode=["bilinear", "nearest", "nearest", "nearest"]
            ),
            Orientationd(keys=["image", "tumor_mask", "oar_mask_0", "oar_mask_1"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-200, a_max=200,
                b_min=0.0, b_max=1.0,
                clip=True
            ),
            Resized(
                keys=["image", "tumor_mask", "oar_mask_0", "oar_mask_1"],
                spatial_size=(128, 128, 128),
                mode=["trilinear", "nearest", "nearest", "nearest"]
            ),
            EnsureTyped(keys=["image", "tumor_mask", "oar_mask_0", "oar_mask_1"])
        ])

        data_dict = {
            "image": ct_path,
            "tumor_mask": tumor_mask_path,
            "oar_mask_0": duodenum_path,
            "oar_mask_1": stomach_path
        }
        data_dict = dose_transform(data_dict)

        # 입력 준비
        inputs = torch.cat([
            data_dict["image"],
            data_dict["tumor_mask"],
            data_dict["oar_mask_0"],
            data_dict["oar_mask_1"]
        ], dim=0).unsqueeze(0).to(self.device)

        prescription_tensor = torch.tensor([prescription_dose], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions = self.dose_model(inputs, prescription_dose=prescription_tensor)

        dose_map = predictions["dose_map"].cpu().squeeze().numpy()
        dose_map = dose_map * prescription_dose

        # Temporary 파일 삭제
        Path(duodenum_path).unlink()
        Path(stomach_path).unlink()

        print(f"    ✅ Dose prediction 완료 (max dose: {dose_map.max():.2f} Gy)")
        return dose_map

    def _predict_outcomes(
        self,
        ct_path: str,
        tumor_mask_path: str,
        clinical_data: Dict
    ) -> Dict:
        """
        Step 3: 생존/독성/반응 예측.

        Args:
            ct_path: CT scan 경로
            tumor_mask_path: Tumor mask 경로
            clinical_data: Clinical features dict

        Returns:
            predictions: {survival_time, toxicity_grade, response, ...}
        """
        print("  [Step 3/3] Multi-task Prediction 실행 중...")

        data_dict = {
            "image": ct_path,
            "tumor_mask": tumor_mask_path
        }
        data_dict = self.multitask_transform(data_dict)

        # Clinical features
        feature_names = [
            'age', 'gender', 'stage', 'ca19_9', 'tumor_size',
            'location', 'kps', 'diabetes', 'prior_surgery', 'chemotherapy'
        ]
        features = [float(clinical_data.get(name, 0.0)) for name in feature_names]
        clinical_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 이미지
        image = data_dict["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.multitask_model(image, clinical=clinical_tensor)

        results = {
            "survival_time": predictions["survival_time"].cpu().item(),
            "survival_uncertainty": predictions["survival_uncertainty"].cpu().item(),
            "toxicity_probs": predictions["toxicity_probs"].cpu().squeeze().tolist(),
            "toxicity_grade": torch.argmax(predictions["toxicity_probs"], dim=1).cpu().item(),
            "response_prob": predictions["response_prob"].cpu().item(),
            "response": predictions["response_prob"].cpu().item() > 0.5
        }

        print(f"    ✅ Outcome prediction 완료")
        print(f"       - Survival: {results['survival_time']:.1f} months")
        print(f"       - Toxicity: Grade {results['toxicity_grade']}")
        print(f"       - Response: {'Responder' if results['response'] else 'Non-responder'}")

        return results

    def run(
        self,
        ct_path: str,
        tumor_mask_path: str,
        clinical_data: Dict,
        prescription_dose: float = 40.0,
        output_dir: Optional[str] = None,
        patient_id: str = "patient_1"
    ) -> Dict:
        """
        전체 파이프라인 실행.

        Args:
            ct_path: CT scan 경로
            tumor_mask_path: Tumor mask 경로
            clinical_data: Clinical features dict
            prescription_dose: 처방 선량 (Gy)
            output_dir: 출력 디렉토리 (None이면 저장 안 함)
            patient_id: 환자 ID

        Returns:
            results: {
                "oar_segmentation": np.ndarray,
                "dose_map": np.ndarray,
                "outcomes": {...}
            }
        """
        print("\n" + "="*60)
        print(f"🏥 Radiotherapy Pipeline 실행: {patient_id}")
        print("="*60 + "\n")

        # Step 1: OAR Segmentation
        oar_mask = self._segment_oars(ct_path)

        # Step 2: Dose Prediction
        dose_map = self._predict_dose(
            ct_path,
            tumor_mask_path,
            oar_mask,
            prescription_dose
        )

        # Step 3: Multi-task Prediction
        outcomes = self._predict_outcomes(
            ct_path,
            tumor_mask_path,
            clinical_data
        )

        results = {
            "patient_id": patient_id,
            "oar_segmentation": oar_mask,
            "dose_map": dose_map,
            "outcomes": outcomes
        }

        # 결과 저장
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # OAR segmentation 저장
            ref_nii = nib.load(ct_path)
            oar_nii = nib.Nifti1Image(oar_mask, ref_nii.affine, ref_nii.header)
            nib.save(oar_nii, output_path / f"{patient_id}_oar_segmentation.nii.gz")

            # Dose map 저장
            dose_nii = nib.Nifti1Image(dose_map, ref_nii.affine, ref_nii.header)
            nib.save(dose_nii, output_path / f"{patient_id}_dose_map.nii.gz")

            # Outcomes 저장 (JSON)
            with open(output_path / f"{patient_id}_outcomes.json", 'w') as f:
                json.dump(outcomes, f, indent=2)

            print(f"\n✅ 결과 저장 완료: {output_path}")

        print("\n" + "="*60)
        print("✅ Pipeline 완료!")
        print("="*60 + "\n")

        return results


def main():
    """CLI 실행"""
    parser = argparse.ArgumentParser(description="End-to-End Radiotherapy Pipeline")

    # 모델 체크포인트
    parser.add_argument("--oar_checkpoint", type=str, required=True,
                        help="OAR segmentation 모델 체크포인트")
    parser.add_argument("--dose_checkpoint", type=str, required=True,
                        help="Dose prediction 모델 체크포인트")
    parser.add_argument("--multitask_checkpoint", type=str, required=True,
                        help="Multi-task 모델 체크포인트")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="디바이스")

    # 입력 데이터
    parser.add_argument("--patient_id", type=str, default="patient_1",
                        help="환자 ID")
    parser.add_argument("--ct_path", type=str, required=True,
                        help="CT scan 경로")
    parser.add_argument("--tumor_mask_path", type=str, required=True,
                        help="Tumor mask 경로")
    parser.add_argument("--clinical_json", type=str, required=True,
                        help='Clinical data (JSON string), e.g., \'{"age": 65, "stage": 2}\'')
    parser.add_argument("--prescription_dose", type=float, default=40.0,
                        help="처방 선량 (Gy)")

    # 출력
    parser.add_argument("--output_dir", type=str, default="outputs/pipeline",
                        help="출력 디렉토리")

    args = parser.parse_args()

    # Pipeline 실행
    pipeline = RadiotherapyPipeline(
        oar_checkpoint=args.oar_checkpoint,
        dose_checkpoint=args.dose_checkpoint,
        multitask_checkpoint=args.multitask_checkpoint,
        device=args.device
    )

    clinical_data = json.loads(args.clinical_json)

    results = pipeline.run(
        ct_path=args.ct_path,
        tumor_mask_path=args.tumor_mask_path,
        clinical_data=clinical_data,
        prescription_dose=args.prescription_dose,
        output_dir=args.output_dir,
        patient_id=args.patient_id
    )


if __name__ == "__main__":
    main()
