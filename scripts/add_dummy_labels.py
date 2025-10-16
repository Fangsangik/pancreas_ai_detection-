
import os
import json
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_dummy_label_if_not_exists(image_path: Path, label_path: Path):
    """가짜 레이블 파일이 없으면, 중앙에 사각형이 있는 레이블을 생성합니다."""
    if label_path.exists():
        # 기존의 잘못된 레이블을 삭제하고 다시 만듭니다.
        logging.info(f"기존 가짜 레이블 삭제: {label_path}")
        os.remove(label_path)

    logging.info(f"가상 췌장 레이블 생성 중: {label_path}")
    try:
        image_nii = nib.load(str(image_path))
        shape = image_nii.shape
        dummy_label_data = np.zeros(shape, dtype=np.uint8)

        # 이미지 중앙에 32x32x32 크기의 가상 췌장 영역(값=1)을 생성합니다.
        center = [s // 2 for s in shape]
        half_size = 16
        
        start = [c - half_size for c in center]
        end = [c + half_size for c in center]
        
        # 인덱스가 이미지 범위를 벗어나지 않도록 조정
        start = [max(0, s) for s in start]
        end = [min(shape[i], e) for i, e in enumerate(end)]

        dummy_label_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = 1

        dummy_label_nii = nib.Nifti1Image(dummy_label_data, image_nii.affine, image_nii.header)
        
        label_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(dummy_label_nii, str(label_path))

    except Exception as e:
        logging.error(f"레이블 파일 생성 실패: {label_path}, 오류: {e}")
        raise

def main(args):
    """메인 실행 함수"""
    manifest_path = Path(args.manifest_file)
    # data_root는 manifest 파일 위치의 부모 디렉토리 (예: data/)
    data_root = manifest_path.parent.parent 

    if not manifest_path.exists():
        logging.error(f"Manifest 파일을 찾을 수 없습니다: {manifest_path}")
        return

    logging.info(f"Manifest 파일 처리 시작: {manifest_path}")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    updated_count = 0

    # 'data' 키 아래의 train, val, test 분할을 모두 순회
    for split in manifest.get('data', {}):
        for item in manifest['data'][split]:
            # if 'label' in item and item['label']:
            #     continue # 이미 레이블이 있으면 건너뛰기 (강제 생성을 위해 주석 처리)

            if 'path' not in item:
                logging.warning(f"'path' 키를 찾을 수 없는 항목을 건너뜁니다: {item}")
                continue

            # 절대 이미지 경로 가져오기
            image_abs_path = Path(item['path'])
            image_filename = image_abs_path.name

            # 레이블 경로 생성
            label_filename = image_filename.replace('_ct.nii.gz', '_seg.nii.gz').replace('.nii.gz', '_seg.nii.gz')
            label_rel_path = Path('labels') / label_filename
            label_abs_path = data_root / label_rel_path

            # 가짜 레이블 파일 생성
            create_dummy_label_if_not_exists(image_abs_path, label_abs_path)

            # manifest 아이템 업데이트 (상대 경로로 저장)
            item['label'] = label_rel_path.as_posix()
            updated_count += 1

    if updated_count > 0:
        # 업데이트된 manifest를 원본 파일에 덮어쓰기
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logging.info(f"✓ Manifest 업데이트 완료. {updated_count}개 항목에 레이블 추가.")
    else:
        logging.info("모든 항목에 이미 레이블이 존재합니다. 변경사항 없음.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='데이터 Manifest에 가짜 레이블을 생성하고 추가합니다.')
    parser.add_argument(
        '--manifest_file',
        type=str,
        default='data/manifests/pancreas_ct_manifest.json',
        help='처리할 메인 Manifest JSON 파일'
    )
    args = parser.parse_args()
    main(args)
