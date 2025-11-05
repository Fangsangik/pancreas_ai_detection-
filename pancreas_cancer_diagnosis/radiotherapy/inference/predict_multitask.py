"""
Multi-Task 모델 추론 스크립트
==============================

Survival + Toxicity + Response 예측
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, \
    Orientationd, ScaleIntensityRanged, Resized, EnsureTyped

from pancreas_cancer_diagnosis.radiotherapy.models import MultiTaskRadiotherapyModel


def get_inference_transform(spatial_size=(96, 96, 96)):
    """추론용 전처리 파이프라인"""
    return Compose([
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
            a_min=-200,
            a_max=200,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        Resized(
            keys=["image", "tumor_mask"],
            spatial_size=spatial_size,
            mode=("trilinear", "nearest")
        ),
        EnsureTyped(keys=["image", "tumor_mask"])
    ])


def load_clinical_features(clinical_dict: Dict) -> torch.Tensor:
    """Clinical features 추출"""
    feature_names = [
        'age', 'gender', 'stage', 'ca19_9', 'tumor_size',
        'location', 'kps', 'diabetes', 'prior_surgery', 'chemotherapy'
    ]

    features = []
    for name in feature_names:
        value = clinical_dict.get(name, 0.0)
        features.append(float(value))

    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, 10)


def predict_single(
    model: MultiTaskRadiotherapyModel,
    ct_path: str,
    tumor_mask_path: str,
    clinical_data: Dict,
    transform,
    device: str = "cuda"
) -> Dict:
    """
    단일 환자 예측.

    Args:
        model: 학습된 모델
        ct_path: CT scan 경로
        tumor_mask_path: Tumor mask 경로
        clinical_data: Clinical features dict
        transform: 전처리 파이프라인
        device: 디바이스

    Returns:
        {
            "survival_time": float (months),
            "survival_uncertainty": float (log variance),
            "toxicity_probs": [p0, p1, p2, p3] (확률 분포),
            "toxicity_grade": int (예측 등급),
            "response_prob": float (responder 확률),
            "response": bool (responder 여부)
        }
    """
    model.eval()

    # 1. 이미지 로딩 및 전처리
    data_dict = {
        "image": ct_path,
        "tumor_mask": tumor_mask_path
    }
    data_dict = transform(data_dict)

    # 2. Clinical features
    clinical_features = load_clinical_features(clinical_data)

    # 3. GPU로 이동
    image = data_dict["image"].unsqueeze(0).to(device)  # (1, 1, D, H, W)
    clinical_features = clinical_features.to(device)

    # 4. 예측
    with torch.no_grad():
        predictions = model(image, clinical=clinical_features)

    # 5. 결과 정리
    results = {
        # Survival
        "survival_time": predictions["survival_time"].cpu().item(),
        "survival_uncertainty": predictions["survival_uncertainty"].cpu().item(),

        # Toxicity
        "toxicity_probs": predictions["toxicity_probs"].cpu().squeeze().tolist(),
        "toxicity_grade": torch.argmax(predictions["toxicity_probs"], dim=1).cpu().item(),

        # Response
        "response_prob": predictions["response_prob"].cpu().item(),
        "response": predictions["response_prob"].cpu().item() > 0.5
    }

    return results


def main(args):
    """메인 추론 함수"""

    print("\n" + "="*60)
    print("Multi-Task 모델 추론")
    print("="*60)

    # 1. 모델 로드
    print("\n1. 모델 로딩 중...")
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"   디바이스: {device}")

    model = MultiTaskRadiotherapyModel.load_from_checkpoint(
        args.checkpoint,
        map_location=device
    )
    model.to(device)
    model.eval()
    print("   ✅ 모델 로드 완료")

    # 2. Transform 준비
    transform = get_inference_transform(spatial_size=tuple(args.spatial_size))

    # 3. 입력 데이터 준비
    print("\n2. 입력 데이터 로딩 중...")
    if args.input_list:
        # JSON 파일에서 여러 환자 로드
        with open(args.input_list, 'r') as f:
            patient_list = json.load(f)
    else:
        # 단일 환자
        patient_list = [{
            "patient_id": args.patient_id or "patient_1",
            "ct_path": args.ct_path,
            "tumor_mask_path": args.tumor_mask_path,
            "clinical": json.loads(args.clinical_json) if args.clinical_json else {}
        }]

    print(f"   총 {len(patient_list)}명의 환자")

    # 4. 추론 실행
    print("\n3. 추론 실행 중...")
    results_list = []

    for idx, patient in enumerate(patient_list):
        print(f"\n[{idx+1}/{len(patient_list)}] {patient['patient_id']}")

        result = predict_single(
            model=model,
            ct_path=patient["ct_path"],
            tumor_mask_path=patient["tumor_mask_path"],
            clinical_data=patient.get("clinical", {}),
            transform=transform,
            device=device
        )

        # 결과 출력
        print(f"   Survival: {result['survival_time']:.1f} months "
              f"(uncertainty: {np.exp(result['survival_uncertainty']):.2f})")
        print(f"   Toxicity: Grade {result['toxicity_grade']} "
              f"(probs: {[f'{p:.3f}' for p in result['toxicity_probs']]})")
        print(f"   Response: {'Responder' if result['response'] else 'Non-responder'} "
              f"(prob: {result['response_prob']:.3f})")

        # 결과 저장
        results_list.append({
            "patient_id": patient["patient_id"],
            **result
        })

    # 5. 결과 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results_list, f, indent=2)

    print("\n" + "="*60)
    print("✅ 추론 완료!")
    print("="*60)
    print(f"결과 저장 위치: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-task model inference")

    # 모델 관련
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="모델 체크포인트 경로")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="디바이스")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[96, 96, 96],
                        help="이미지 크기 (D H W)")

    # 입력 데이터 (Option 1: 단일 환자)
    parser.add_argument("--patient_id", type=str,
                        help="환자 ID")
    parser.add_argument("--ct_path", type=str,
                        help="CT scan 경로")
    parser.add_argument("--tumor_mask_path", type=str,
                        help="Tumor mask 경로")
    parser.add_argument("--clinical_json", type=str,
                        help='Clinical data (JSON string), e.g., \'{"age": 65, "stage": 2}\'')

    # 입력 데이터 (Option 2: 여러 환자)
    parser.add_argument("--input_list", type=str,
                        help="입력 JSON 파일 (여러 환자)")

    # 출력
    parser.add_argument("--output", type=str, default="outputs/multitask_predictions.json",
                        help="출력 JSON 파일")

    args = parser.parse_args()

    # 입력 검증
    if not args.input_list and not (args.ct_path and args.tumor_mask_path):
        parser.error("--input_list 또는 (--ct_path + --tumor_mask_path) 중 하나는 필수입니다.")

    main(args)
