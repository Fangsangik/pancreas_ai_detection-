#!/usr/bin/env python3
"""
TCIA 데이터 다운로드 스크립트
============================

TCIA (The Cancer Imaging Archive) 매니페스트 파일을 사용하여
췌장 CT 데이터를 다운로드합니다.

필요 사항:
- NBIA Data Retriever 설치 필요
- 또는 tcia-utils Python 패키지 사용

사용법:
    python download_tcia_data.py --manifest Pancreas-CT-20200910.tcia --output ./data/raw
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess
import shutil


def check_nbia_retriever():
    """
    NBIA Data Retriever가 설치되어 있는지 확인

    Returns:
        str or None: NBIA 실행 파일 경로, 없으면 None
    """
    # macOS 기본 설치 경로
    mac_paths = [
        "/Applications/NBIA Data Retriever.app/Contents/MacOS/NBIA Data Retriever",
        os.path.expanduser("~/Applications/NBIA Data Retriever.app/Contents/MacOS/NBIA Data Retriever")
    ]

    for path in mac_paths:
        if os.path.exists(path):
            return path

    # 명령줄에서 찾기
    nbia_cmd = shutil.which("NBIADataRetriever")
    if nbia_cmd:
        return nbia_cmd

    return None


def download_with_nbia(manifest_path: str, output_dir: str):
    """
    NBIA Data Retriever를 사용하여 데이터 다운로드

    Args:
        manifest_path (str): .tcia 매니페스트 파일 경로
        output_dir (str): 다운로드 출력 디렉토리
    """
    nbia_path = check_nbia_retriever()

    if not nbia_path:
        print("❌ NBIA Data Retriever를 찾을 수 없습니다.")
        print("\n다운로드 방법:")
        print("1. https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images 방문")
        print("2. NBIA Data Retriever 다운로드 및 설치")
        print("3. 이 스크립트 다시 실행")
        sys.exit(1)

    print(f"✓ NBIA Data Retriever 발견: {nbia_path}")

    # NBIA로 다운로드
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📦 데이터 다운로드 시작...")
    print(f"   매니페스트: {manifest_path}")
    print(f"   출력 디렉토리: {output_dir}")

    # NBIA 실행
    # 주의: NBIA는 GUI 앱이므로 자동화가 어려울 수 있음
    print("\n⚠️  NBIA Data Retriever GUI가 열립니다.")
    print("   1. 매니페스트 파일 선택")
    print("   2. 다운로드 위치 선택")
    print("   3. 다운로드 시작")

    subprocess.run([nbia_path, manifest_path])


def download_with_tcia_utils(manifest_path: str, output_dir: str):
    """
    tcia-utils Python 패키지를 사용한 다운로드 (대안)

    Args:
        manifest_path (str): .tcia 매니페스트 파일 경로
        output_dir (str): 다운로드 출력 디렉토리
    """
    try:
        from tcia_utils import nbia
    except ImportError:
        print("❌ tcia-utils 패키지가 설치되지 않았습니다.")
        print("\n설치 방법:")
        print("   pip install tcia-utils")
        sys.exit(1)

    print("📦 tcia-utils를 사용하여 데이터 다운로드 중...")

    # 매니페스트 파일 파싱
    with open(manifest_path, 'r') as f:
        lines = f.readlines()

    # Series UID 추출
    series_uids = []
    in_list = False
    for line in lines:
        line = line.strip()
        if line.startswith("ListOfSeriesToDownload="):
            in_list = True
            continue
        if in_list and line and not line.startswith("#"):
            series_uids.append(line)

    print(f"✓ {len(series_uids)}개의 시리즈를 다운로드합니다.")

    os.makedirs(output_dir, exist_ok=True)

    # 각 시리즈 다운로드
    for i, series_uid in enumerate(series_uids, 1):
        print(f"\n[{i}/{len(series_uids)}] 다운로드 중: {series_uid}")
        try:
            # NBIA API로 다운로드
            nbia.downloadSeries(
                series_uid=series_uid,
                input_type="uid",
                path=output_dir
            )
            print(f"   ✓ 완료")
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            continue


def print_manual_instructions(manifest_path: str):
    """
    수동 다운로드 방법 안내

    Args:
        manifest_path (str): .tcia 매니페스트 파일 경로
    """
    print("=" * 70)
    print("📖 TCIA 데이터 수동 다운로드 방법")
    print("=" * 70)
    print()
    print("1️⃣  NBIA Data Retriever 다운로드:")
    print("   https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images")
    print()
    print("2️⃣  설치 후 애플리케이션 실행")
    print()
    print("3️⃣  'Browse' 버튼 클릭하여 매니페스트 파일 선택:")
    print(f"   {manifest_path}")
    print()
    print("4️⃣  다운로드 위치 선택 및 다운로드 시작")
    print()
    print("5️⃣  다운로드 완료 후, DICOM 파일들을 NIfTI로 변환:")
    print("   python scripts/convert_dicom_to_nifti.py --input ./data/raw --output ./data/nifti")
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="TCIA 췌장 CT 데이터 다운로드"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="/Users/hwangsang-ik/Downloads/Pancreas-CT-20200910.tcia",
        help="TCIA 매니페스트 파일 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/raw_dicom",
        help="다운로드 출력 디렉토리"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["nbia", "tcia-utils", "manual"],
        default="manual",
        help="다운로드 방법 선택"
    )

    args = parser.parse_args()

    # 매니페스트 파일 확인
    if not os.path.exists(args.manifest):
        print(f"❌ 매니페스트 파일을 찾을 수 없습니다: {args.manifest}")
        sys.exit(1)

    manifest_path = os.path.abspath(args.manifest)
    output_dir = os.path.abspath(args.output)

    print("=" * 70)
    print("TCIA 췌장 CT 데이터 다운로드")
    print("=" * 70)
    print()

    if args.method == "nbia":
        download_with_nbia(manifest_path, output_dir)
    elif args.method == "tcia-utils":
        download_with_tcia_utils(manifest_path, output_dir)
    else:  # manual
        print_manual_instructions(manifest_path)


if __name__ == "__main__":
    main()
