#!/usr/bin/env python3
"""
DICOM to NIfTI 변환 스크립트
===========================

TCIA에서 다운로드한 DICOM 파일들을 NIfTI 포맷으로 변환합니다.

필요 패키지:
- SimpleITK 또는 pydicom + nibabel

사용법:
    python convert_dicom_to_nifti.py --input ./data/raw_dicom --output ./data/nifti
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import json


def check_dependencies():
    """필요한 패키지 확인"""
    try:
        import SimpleITK as sitk
        return "sitk"
    except ImportError:
        pass

    try:
        import pydicom
        import nibabel as nib
        import numpy as np
        return "pydicom"
    except ImportError:
        print("❌ 필요한 패키지가 설치되지 않았습니다.")
        print("\n다음 중 하나를 설치하세요:")
        print("   1. pip install SimpleITK  (권장)")
        print("   2. pip install pydicom nibabel")
        sys.exit(1)


def find_dicom_series(root_dir: str) -> List[str]:
    """
    DICOM 시리즈 디렉토리 찾기

    Args:
        root_dir (str): 루트 디렉토리

    Returns:
        List[str]: DICOM 시리즈 디렉토리 경로 리스트
    """
    series_dirs = []

    for root, dirs, files in os.walk(root_dir):
        # DICOM 파일이 있는 디렉토리 찾기
        dicom_files = [f for f in files if f.lower().endswith('.dcm') or not '.' in f]
        if dicom_files:
            series_dirs.append(root)

    return series_dirs


def convert_with_sitk(dicom_dir: str, output_path: str) -> bool:
    """
    SimpleITK를 사용한 DICOM to NIfTI 변환

    Args:
        dicom_dir (str): DICOM 시리즈 디렉토리
        output_path (str): 출력 NIfTI 파일 경로

    Returns:
        bool: 성공 여부
    """
    import SimpleITK as sitk

    try:
        # DICOM 시리즈 읽기
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)

        if not dicom_names:
            print(f"   ⚠️  DICOM 파일을 찾을 수 없습니다")
            return False

        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # NIfTI로 저장
        sitk.WriteImage(image, output_path)

        # 메타데이터 저장
        metadata = {
            "size": image.GetSize(),
            "spacing": image.GetSpacing(),
            "origin": image.GetOrigin(),
            "direction": image.GetDirection(),
            "num_slices": len(dicom_names)
        }

        metadata_path = output_path.replace(".nii.gz", "_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return True

    except Exception as e:
        print(f"   ❌ 오류: {e}")
        return False


def convert_with_pydicom(dicom_dir: str, output_path: str) -> bool:
    """
    pydicom + nibabel을 사용한 DICOM to NIfTI 변환

    Args:
        dicom_dir (str): DICOM 시리즈 디렉토리
        output_path (str): 출력 NIfTI 파일 경로

    Returns:
        bool: 성공 여부
    """
    import pydicom
    import nibabel as nib
    import numpy as np

    try:
        # DICOM 파일 리스트
        dicom_files = []
        for f in os.listdir(dicom_dir):
            if f.lower().endswith('.dcm') or not '.' in f:
                dicom_files.append(os.path.join(dicom_dir, f))

        if not dicom_files:
            print(f"   ⚠️  DICOM 파일을 찾을 수 없습니다")
            return False

        # 첫 번째 파일로 메타데이터 읽기
        ref_dicom = pydicom.dcmread(dicom_files[0])

        # 모든 슬라이스 읽기 및 정렬
        slices = []
        for dcm_file in dicom_files:
            ds = pydicom.dcmread(dcm_file)
            slices.append((float(ds.ImagePositionPatient[2]), ds))

        slices.sort(key=lambda x: x[0])

        # 3D 볼륨 생성
        img_shape = list(slices[0][1].pixel_array.shape)
        img_shape.append(len(slices))
        volume = np.zeros(img_shape, dtype=np.float32)

        for i, (_, ds) in enumerate(slices):
            volume[:, :, i] = ds.pixel_array.astype(np.float32)

        # Spacing 정보
        pixel_spacing = ref_dicom.PixelSpacing
        slice_thickness = ref_dicom.SliceThickness if hasattr(ref_dicom, 'SliceThickness') else 1.0

        # NIfTI 생성 및 저장
        nifti_img = nib.Nifti1Image(volume, affine=np.eye(4))
        nifti_img.header.set_zooms([float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness)])

        nib.save(nifti_img, output_path)

        # 메타데이터 저장
        metadata = {
            "size": img_shape,
            "spacing": [float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness)],
            "num_slices": len(slices),
            "modality": str(ref_dicom.Modality) if hasattr(ref_dicom, 'Modality') else 'CT'
        }

        metadata_path = output_path.replace(".nii.gz", "_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return True

    except Exception as e:
        print(f"   ❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="DICOM to NIfTI 변환"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="DICOM 파일이 있는 루트 디렉토리"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="NIfTI 파일 출력 디렉토리"
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="특정 환자 ID만 변환 (선택사항)"
    )

    args = parser.parse_args()

    # 의존성 확인
    method = check_dependencies()
    print(f"✓ 변환 방법: {method}")

    # 입력 디렉토리 확인
    if not os.path.exists(args.input):
        print(f"❌ 입력 디렉토리를 찾을 수 없습니다: {args.input}")
        sys.exit(1)

    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)

    print(f"\n📂 입력: {args.input}")
    print(f"📂 출력: {args.output}")

    # DICOM 시리즈 찾기
    print("\n🔍 DICOM 시리즈 검색 중...")
    series_dirs = find_dicom_series(args.input)
    print(f"✓ {len(series_dirs)}개의 시리즈를 찾았습니다.")

    if not series_dirs:
        print("❌ DICOM 파일을 찾을 수 없습니다.")
        sys.exit(1)

    # 변환 함수 선택
    convert_func = convert_with_sitk if method == "sitk" else convert_with_pydicom

    # 각 시리즈 변환
    success_count = 0
    fail_count = 0

    print("\n🔄 변환 시작...\n")

    for i, series_dir in enumerate(series_dirs, 1):
        # 환자 ID 추출
        parts = Path(series_dir).parts
        patient_id = None
        for part in parts:
            if part.startswith("PANCREAS_"):
                patient_id = part
                break

        if not patient_id:
            patient_id = f"patient_{i:04d}"

        # 특정 환자만 처리
        if args.patient_id and patient_id != args.patient_id:
            continue

        print(f"[{i}/{len(series_dirs)}] {patient_id}")
        print(f"   소스: {series_dir}")

        # 출력 파일명
        output_filename = f"{patient_id}_ct.nii.gz"
        output_path = os.path.join(args.output, output_filename)

        # 이미 존재하면 스킵
        if os.path.exists(output_path):
            print(f"   ⏭️  이미 존재함 (스킵)")
            success_count += 1
            continue

        # 변환
        if convert_func(series_dir, output_path):
            print(f"   ✓ 완료: {output_filename}")
            success_count += 1
        else:
            fail_count += 1

        print()

    # 결과 요약
    print("=" * 70)
    print("변환 완료")
    print("=" * 70)
    print(f"✓ 성공: {success_count}")
    print(f"❌ 실패: {fail_count}")
    print(f"📂 출력 디렉토리: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
