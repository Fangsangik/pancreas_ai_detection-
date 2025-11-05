"""
OAR Segmentation 모델 추론 스크립트
===================================

Organs at Risk 자동 세그멘테이션
"""

import argparse
import json
import torch
import nibabel as nib
import numpy as np
from pathlib import Path

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, \
    Orientationd, ScaleIntensityRanged, Resized, EnsureTyped

from pancreas_cancer_diagnosis.radiotherapy.models import OARSegmentationModel


def get_inference_transform(spatial_size=(128, 128, 128)):
    """추론용 전처리 파이프라인"""
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 1.5),
            mode="bilinear"
        ),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200,
            a_max=200,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        Resized(
            keys=["image"],
            spatial_size=spatial_size,
            mode="trilinear"
        ),
        EnsureTyped(keys=["image"])
    ])


def predict_single(
    model: OARSegmentationModel,
    ct_path: str,
    transform,
    device: str = "cuda"
) -> np.ndarray:
    """
    단일 환자 OAR segmentation.

    Args:
        model: 학습된 모델
        ct_path: CT scan 경로
        transform: 전처리 파이프라인
        device: 디바이스

    Returns:
        oar_mask: (D, H, W) numpy array (class indices 0-6)
    """
    model.eval()

    # 1. 이미지 로딩 및 전처리
    data_dict = {"image": ct_path}
    data_dict = transform(data_dict)

    # 2. GPU로 이동
    image = data_dict["image"].unsqueeze(0).to(device)  # (1, 1, D, H, W)

    # 3. 예측
    with torch.no_grad():
        predictions = model(image)

    # 4. Argmax (multi-class segmentation)
    seg_probs = predictions["seg_probs"]  # (1, 7, D, H, W)
    seg_pred = torch.argmax(seg_probs, dim=1)  # (1, D, H, W)
    oar_mask = seg_pred.cpu().squeeze().numpy().astype(np.uint8)  # (D, H, W)

    return oar_mask


def compute_dice_per_organ(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int = 7):
    """
    각 OAR별 Dice score 계산.

    Args:
        pred_mask: Predicted segmentation (D, H, W)
        gt_mask: Ground truth segmentation (D, H, W)
        num_classes: 클래스 수

    Returns:
        dice_scores: Dict[organ_name, dice]
    """
    oar_names = [
        'background', 'duodenum', 'stomach', 'small_intestine',
        'liver', 'left_kidney', 'right_kidney'
    ]

    dice_scores = {}
    for c in range(1, num_classes):  # Skip background
        pred_c = (pred_mask == c)
        gt_c = (gt_mask == c)

        intersection = np.sum(pred_c & gt_c)
        union = np.sum(pred_c) + np.sum(gt_c)

        if union > 0:
            dice = 2.0 * intersection / union
        else:
            dice = 0.0

        dice_scores[oar_names[c]] = dice

    return dice_scores


def save_segmentation_nifti(seg_mask: np.ndarray, output_path: str, reference_nifti: str = None):
    """
    Segmentation mask를 NIfTI 파일로 저장.

    Args:
        seg_mask: (D, H, W) numpy array
        output_path: 출력 경로
        reference_nifti: 참조 NIfTI (affine, header 복사)
    """
    if reference_nifti:
        ref_nii = nib.load(reference_nifti)
        affine = ref_nii.affine
        header = ref_nii.header
    else:
        affine = np.eye(4)
        header = None

    seg_nii = nib.Nifti1Image(seg_mask, affine=affine, header=header)
    nib.save(seg_nii, output_path)


def main(args):
    """메인 추론 함수"""

    print("\n" + "="*60)
    print("OAR Segmentation 모델 추론")
    print("="*60)

    # 1. 모델 로드
    print("\n1. 모델 로딩 중...")
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"   디바이스: {device}")

    model = OARSegmentationModel.load_from_checkpoint(
        args.checkpoint,
        map_location=device
    )
    model.to(device)
    model.eval()
    print("   ✅ 모델 로드 완료")

    # OAR 이름 출력
    print("\n   세그멘테이션 대상 OARs:")
    for idx, name in enumerate(model.OAR_NAMES):
        print(f"     Class {idx}: {name}")

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
            "gt_oar_mask_path": args.gt_oar_mask_path  # Optional (for evaluation)
        }]

    print(f"   총 {len(patient_list)}명의 환자")

    # 4. 추론 실행
    print("\n3. 추론 실행 중...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_list = []

    for idx, patient in enumerate(patient_list):
        print(f"\n[{idx+1}/{len(patient_list)}] {patient['patient_id']}")

        oar_mask = predict_single(
            model=model,
            ct_path=patient["ct_path"],
            transform=transform,
            device=device
        )

        # 통계 출력
        print(f"   Segmentation shape: {oar_mask.shape}")
        print(f"   Unique classes: {np.unique(oar_mask)}")

        # 각 클래스별 voxel 수
        for c in range(7):
            count = np.sum(oar_mask == c)
            if count > 0:
                print(f"     Class {c} ({model.OAR_NAMES[c]}): {count:,} voxels")

        # Ground truth가 있으면 Dice 계산
        result_dict = {"patient_id": patient["patient_id"]}
        if patient.get("gt_oar_mask_path"):
            gt_mask = nib.load(patient["gt_oar_mask_path"]).get_fdata().astype(np.uint8)
            dice_scores = compute_dice_per_organ(oar_mask, gt_mask)

            print(f"\n   Dice scores:")
            for organ, dice in dice_scores.items():
                print(f"     {organ}: {dice:.4f}")

            result_dict["dice_scores"] = dice_scores
            result_dict["mean_dice"] = np.mean(list(dice_scores.values()))

        results_list.append(result_dict)

        # 저장
        output_path = output_dir / f"{patient['patient_id']}_oar_segmentation.nii.gz"
        save_segmentation_nifti(
            oar_mask,
            str(output_path),
            reference_nifti=patient["ct_path"]
        )
        print(f"\n   ✅ 저장: {output_path}")

    # 5. 결과 저장 (JSON)
    if args.save_metrics:
        metrics_path = output_dir / "segmentation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(results_list, f, indent=2)
        print(f"\n✅ 메트릭 저장: {metrics_path}")

    print("\n" + "="*60)
    print("✅ 추론 완료!")
    print("="*60)
    print(f"결과 저장 위치: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OAR segmentation model inference")

    # 모델 관련
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="모델 체크포인트 경로")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="디바이스")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[128, 128, 128],
                        help="이미지 크기 (D H W)")

    # 입력 데이터 (Option 1: 단일 환자)
    parser.add_argument("--patient_id", type=str,
                        help="환자 ID")
    parser.add_argument("--ct_path", type=str,
                        help="CT scan 경로")
    parser.add_argument("--gt_oar_mask_path", type=str,
                        help="Ground truth OAR mask 경로 (optional, for evaluation)")

    # 입력 데이터 (Option 2: 여러 환자)
    parser.add_argument("--input_list", type=str,
                        help="입력 JSON 파일 (여러 환자)")

    # 출력
    parser.add_argument("--output_dir", type=str, default="outputs/oar_segmentations",
                        help="출력 디렉토리")
    parser.add_argument("--save_metrics", action="store_true",
                        help="메트릭을 JSON으로 저장")

    args = parser.parse_args()

    # 입력 검증
    if not args.input_list and not args.ct_path:
        parser.error("--input_list 또는 --ct_path 중 하나는 필수입니다.")

    main(args)
