#!/usr/bin/env python3
"""
췌장 CT 데이터 준비 스크립트
==========================

NIfTI 포맷으로 변환된 CT 데이터를 검증하고,
학습/검증/테스트 세트로 분할합니다.

사용법:
    python prepare_pancreas_data.py --input ./data/nifti --output ./data/processed
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np


def check_dependencies():
    """필요한 패키지 확인"""
    try:
        import nibabel as nib
        return True
    except ImportError:
        print("❌ nibabel 패키지가 설치되지 않았습니다.")
        print("   pip install nibabel")
        sys.exit(1)


def load_nifti_info(nifti_path: str) -> Dict:
    """
    NIfTI 파일 정보 로드

    Args:
        nifti_path (str): NIfTI 파일 경로

    Returns:
        Dict: 파일 정보
    """
    import nibabel as nib

    try:
        img = nib.load(nifti_path)
        data = img.get_fdata()

        return {
            "path": nifti_path,
            "shape": data.shape,
            "spacing": tuple(float(v) for v in img.header.get_zooms()),
            "dtype": str(data.dtype),
            "min_value": float(np.min(data)),
            "max_value": float(np.max(data)),
            "mean_value": float(np.mean(data)),
            "std_value": float(np.std(data)),
            "valid": True,
            "error": None
        }
    except Exception as e:
        return {
            "path": nifti_path,
            "valid": False,
            "error": str(e)
        }


def validate_ct_data(nifti_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """
    CT 데이터 검증

    Args:
        nifti_dir (str): NIfTI 파일 디렉토리

    Returns:
        Tuple[List[Dict], List[Dict]]: (유효한 파일들, 오류 파일들)
    """
    print("🔍 CT 데이터 검증 중...")

    valid_files = []
    invalid_files = []

    nifti_files = list(Path(nifti_dir).glob("*.nii.gz"))

    if not nifti_files:
        print(f"❌ NIfTI 파일을 찾을 수 없습니다: {nifti_dir}")
        return [], []

    print(f"   총 {len(nifti_files)}개 파일 발견")

    for i, nifti_file in enumerate(nifti_files, 1):
        print(f"   [{i}/{len(nifti_files)}] {nifti_file.name}...", end=" ")

        info = load_nifti_info(str(nifti_file))

        if info["valid"]:
            valid_files.append(info)
            print("✓")
        else:
            invalid_files.append(info)
            print(f"❌ {info['error']}")

    print(f"\n✓ 유효한 파일: {len(valid_files)}")
    print(f"❌ 오류 파일: {len(invalid_files)}")

    return valid_files, invalid_files


def print_data_statistics(valid_files: List[Dict]):
    """
    데이터 통계 출력

    Args:
        valid_files (List[Dict]): 유효한 파일 정보 리스트
    """
    if not valid_files:
        return

    print("\n📊 데이터 통계")
    print("=" * 70)

    # Shape 통계
    shapes = [f["shape"] for f in valid_files]
    print(f"\n영상 크기:")
    for shape in set(map(tuple, shapes)):
        count = shapes.count(list(shape))
        print(f"   {shape}: {count}개")

    # Spacing 통계
    spacings = [f["spacing"] for f in valid_files]
    print(f"\n복셀 간격 (spacing):")
    avg_spacing = np.mean(spacings, axis=0)
    print(f"   평균: ({avg_spacing[0]:.2f}, {avg_spacing[1]:.2f}, {avg_spacing[2]:.2f}) mm")

    # 강도 통계
    min_values = [f["min_value"] for f in valid_files]
    max_values = [f["max_value"] for f in valid_files]
    mean_values = [f["mean_value"] for f in valid_files]

    print(f"\n강도 값 범위:")
    print(f"   최소값 범위: [{np.min(min_values):.1f}, {np.max(min_values):.1f}]")
    print(f"   최대값 범위: [{np.min(max_values):.1f}, {np.max(max_values):.1f}]")
    print(f"   평균값 범위: [{np.min(mean_values):.1f}, {np.max(mean_values):.1f}]")

    print("=" * 70)


def create_data_split(
    valid_files: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, List[Dict]]:
    """
    데이터를 학습/검증/테스트 세트로 분할

    Args:
        valid_files (List[Dict]): 유효한 파일 정보 리스트
        train_ratio (float): 학습 세트 비율
        val_ratio (float): 검증 세트 비율
        test_ratio (float): 테스트 세트 비율
        random_seed (int): 랜덤 시드

    Returns:
        Dict[str, List[Dict]]: 분할된 데이터
    """
    np.random.seed(random_seed)

    # 파일 리스트 셔플
    indices = np.random.permutation(len(valid_files))

    # 분할 인덱스 계산
    n_train = int(len(valid_files) * train_ratio)
    n_val = int(len(valid_files) * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    return {
        "train": [valid_files[i] for i in train_indices],
        "val": [valid_files[i] for i in val_indices],
        "test": [valid_files[i] for i in test_indices]
    }


def save_data_manifest(
    data_split: Dict[str, List[Dict]],
    output_dir: str,
    dataset_name: str = "pancreas_ct"
):
    """
    데이터 매니페스트 저장

    Args:
        data_split (Dict): 분할된 데이터
        output_dir (str): 출력 디렉토리
        dataset_name (str): 데이터셋 이름
    """
    os.makedirs(output_dir, exist_ok=True)

    # 전체 매니페스트
    manifest = {
        "dataset_name": dataset_name,
        "total_samples": sum(len(v) for v in data_split.values()),
        "splits": {
            k: len(v) for k, v in data_split.items()
        },
        "data": data_split
    }

    manifest_path = os.path.join(output_dir, f"{dataset_name}_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ 매니페스트 저장: {manifest_path}")

    # 각 split별 파일 리스트 저장 (PyTorch 데이터 로더용)
    for split_name, files in data_split.items():
        split_list = []
        for file_info in files:
            # 상대 경로로 변환
            rel_path = Path(file_info["path"]).name

            split_list.append({
                "image": rel_path,
                # 나중에 세그멘테이션 레이블 경로 추가 가능
                # "label": rel_path.replace("_ct.nii.gz", "_seg.nii.gz")
            })

        list_path = os.path.join(output_dir, f"{dataset_name}_{split_name}.json")
        with open(list_path, 'w') as f:
            json.dump(split_list, f, indent=2)

        print(f"✓ {split_name} 리스트 저장: {list_path} ({len(split_list)}개)")


def main():
    parser = argparse.ArgumentParser(
        description="췌장 CT 데이터 준비 및 검증"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="NIfTI 파일 디렉토리"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="출력 디렉토리 (매니페스트 저장)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="학습 세트 비율 (기본값: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="검증 세트 비율 (기본값: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="테스트 세트 비율 (기본값: 0.15)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="랜덤 시드 (기본값: 42)"
    )

    args = parser.parse_args()

    # 의존성 확인
    check_dependencies()

    print("=" * 70)
    print("췌장 CT 데이터 준비")
    print("=" * 70)
    print()

    # 입력 디렉토리 확인
    if not os.path.exists(args.input):
        print(f"❌ 입력 디렉토리를 찾을 수 없습니다: {args.input}")
        sys.exit(1)

    # 데이터 검증
    valid_files, invalid_files = validate_ct_data(args.input)

    if not valid_files:
        print("❌ 유효한 CT 데이터가 없습니다.")
        sys.exit(1)

    # 통계 출력
    print_data_statistics(valid_files)

    # 데이터 분할
    print("\n📂 데이터 분할 중...")
    data_split = create_data_split(
        valid_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )

    print(f"   학습: {len(data_split['train'])}개")
    print(f"   검증: {len(data_split['val'])}개")
    print(f"   테스트: {len(data_split['test'])}개")

    # 매니페스트 저장
    print("\n💾 매니페스트 저장 중...")
    save_data_manifest(data_split, args.output)

    print("\n" + "=" * 70)
    print("✓ 데이터 준비 완료!")
    print("=" * 70)
    print("\n다음 단계:")
    print("1. 세그멘테이션 레이블 준비 (수동 어노테이션 또는 기존 레이블)")
    print("2. 데이터 매니페스트에 레이블 경로 추가")
    print("3. 학습 시작:")
    print(f"   python pancreas_cancer_diagnosis/segmentation/training/train.py \\")
    print(f"       --data-root {args.input} \\")
    print(f"       --train-list {args.output}/pancreas_ct_train.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
