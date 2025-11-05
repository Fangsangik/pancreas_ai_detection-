"""
Dose Prediction 모델 추론 스크립트
==================================

3D 선량 분포 예측
"""

import argparse
import json
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import List

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, \
    Orientationd, ScaleIntensityRanged, Resized, EnsureTyped

from pancreas_cancer_diagnosis.radiotherapy.models import DosePredictionModel


def get_inference_transform(spatial_size=(128, 128, 128), num_oars=2):
    """추론용 전처리 파이프라인"""
    keys = ["image", "tumor_mask"] + [f"oar_mask_{i}" for i in range(num_oars)]

    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Spacingd(
            keys=keys,
            pixdim=(2.0, 2.0, 2.0),
            mode=["bilinear"] + ["nearest"] * (len(keys) - 1)
        ),
        Orientationd(keys=keys, axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200,
            a_max=200,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        Resized(
            keys=keys,
            spatial_size=spatial_size,
            mode=["trilinear"] + ["nearest"] * (len(keys) - 1)
        ),
        EnsureTyped(keys=keys)
    ])


def predict_single(
    model: DosePredictionModel,
    ct_path: str,
    tumor_mask_path: str,
    oar_mask_paths: List[str],
    prescription_dose: float,
    transform,
    device: str = "cuda"
) -> np.ndarray:
    """
    단일 환자 dose prediction.

    Args:
        model: 학습된 모델
        ct_path: CT scan 경로
        tumor_mask_path: Tumor mask 경로
        oar_mask_paths: OAR masks 경로 리스트
        prescription_dose: 처방 선량 (Gy)
        transform: 전처리 파이프라인
        device: 디바이스

    Returns:
        dose_map: (D, H, W) numpy array (Gy 단위)
    """
    model.eval()

    # 1. 이미지 로딩 및 전처리
    data_dict = {
        "image": ct_path,
        "tumor_mask": tumor_mask_path
    }
    for i, oar_path in enumerate(oar_mask_paths):
        data_dict[f"oar_mask_{i}"] = oar_path

    data_dict = transform(data_dict)

    # 2. 입력 준비 (CT + tumor + OARs를 채널 방향으로 concat)
    inputs = [data_dict["image"]]  # (1, D, H, W)
    inputs.append(data_dict["tumor_mask"])

    for i in range(len(oar_mask_paths)):
        inputs.append(data_dict[f"oar_mask_{i}"])

    input_tensor = torch.cat(inputs, dim=0).unsqueeze(0)  # (1, C, D, H, W)
    input_tensor = input_tensor.to(device)

    # Prescription dose
    prescription_tensor = torch.tensor([prescription_dose], dtype=torch.float32).to(device)

    # 3. 예측
    with torch.no_grad():
        predictions = model(input_tensor, prescription_dose=prescription_tensor)

    # 4. Dose map 추출 (0-1 -> Gy)
    dose_map = predictions["dose_map"].cpu().squeeze().numpy()  # (D, H, W)
    dose_map = dose_map * prescription_dose  # Denormalize

    return dose_map


def save_dose_nifti(dose_map: np.ndarray, output_path: str, reference_nifti: str = None):
    """
    Dose map을 NIfTI 파일로 저장.

    Args:
        dose_map: (D, H, W) numpy array
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

    dose_nii = nib.Nifti1Image(dose_map, affine=affine, header=header)
    nib.save(dose_nii, output_path)


def main(args):
    """메인 추론 함수"""

    print("\n" + "="*60)
    print("Dose Prediction 모델 추론")
    print("="*60)

    # 1. 모델 로드
    print("\n1. 모델 로딩 중...")
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"   디바이스: {device}")

    model = DosePredictionModel.load_from_checkpoint(
        args.checkpoint,
        map_location=device
    )
    model.to(device)
    model.eval()
    print("   ✅ 모델 로드 완료")

    # 2. Transform 준비
    transform = get_inference_transform(
        spatial_size=tuple(args.spatial_size),
        num_oars=args.num_oars
    )

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
            "oar_mask_paths": args.oar_mask_paths,
            "prescription_dose": args.prescription_dose
        }]

    print(f"   총 {len(patient_list)}명의 환자")

    # 4. 추론 실행
    print("\n3. 추론 실행 중...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, patient in enumerate(patient_list):
        print(f"\n[{idx+1}/{len(patient_list)}] {patient['patient_id']}")

        dose_map = predict_single(
            model=model,
            ct_path=patient["ct_path"],
            tumor_mask_path=patient["tumor_mask_path"],
            oar_mask_paths=patient["oar_mask_paths"][:args.num_oars],
            prescription_dose=patient["prescription_dose"],
            transform=transform,
            device=device
        )

        # 통계 출력
        print(f"   Dose map shape: {dose_map.shape}")
        print(f"   Min dose: {dose_map.min():.2f} Gy")
        print(f"   Max dose: {dose_map.max():.2f} Gy")
        print(f"   Mean dose: {dose_map.mean():.2f} Gy")

        # 저장
        output_path = output_dir / f"{patient['patient_id']}_dose.nii.gz"
        save_dose_nifti(
            dose_map,
            str(output_path),
            reference_nifti=patient["ct_path"]
        )
        print(f"   ✅ 저장: {output_path}")

    print("\n" + "="*60)
    print("✅ 추론 완료!")
    print("="*60)
    print(f"결과 저장 위치: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dose prediction model inference")

    # 모델 관련
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="모델 체크포인트 경로")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="디바이스")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[128, 128, 128],
                        help="이미지 크기 (D H W)")
    parser.add_argument("--num_oars", type=int, default=2,
                        help="OAR 개수")

    # 입력 데이터 (Option 1: 단일 환자)
    parser.add_argument("--patient_id", type=str,
                        help="환자 ID")
    parser.add_argument("--ct_path", type=str,
                        help="CT scan 경로")
    parser.add_argument("--tumor_mask_path", type=str,
                        help="Tumor mask 경로")
    parser.add_argument("--oar_mask_paths", type=str, nargs='+',
                        help="OAR masks 경로 (공백으로 구분)")
    parser.add_argument("--prescription_dose", type=float,
                        help="처방 선량 (Gy)")

    # 입력 데이터 (Option 2: 여러 환자)
    parser.add_argument("--input_list", type=str,
                        help="입력 JSON 파일 (여러 환자)")

    # 출력
    parser.add_argument("--output_dir", type=str, default="outputs/dose_predictions",
                        help="출력 디렉토리")

    args = parser.parse_args()

    # 입력 검증
    if not args.input_list and not (args.ct_path and args.tumor_mask_path and args.oar_mask_paths):
        parser.error("--input_list 또는 (--ct_path + --tumor_mask_path + --oar_mask_paths) 중 하나는 필수입니다.")

    main(args)
