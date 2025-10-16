
import os
import json
import numpy as np
import SimpleITK as sitk
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 설정 ---
# 기본 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
NIFTI_DIR = os.path.join(DATA_DIR, 'nifti')
LABELS_DIR = os.path.join(DATA_DIR, 'labels')
MANIFESTS_DIR = os.path.join(DATA_DIR, 'manifests')

# 생성할 더미 데이터 개수
NUM_TRAIN = 3
NUM_VAL = 1
NUM_TEST = 1

# 더미 이미지 속성
IMAGE_SHAPE = (64, 64, 64)  # (Depth, Height, Width)
IMAGE_SPACING = (1.0, 1.0, 1.0)

# --- 함수 정의 ---

def create_dummy_nifti(file_path: str, is_label: bool):
    """지정된 경로에 더미 NIfTI 파일을 생성합니다."""
    try:
        if os.path.exists(file_path):
            logging.info(f"파일이 이미 존재합니다: {file_path}")
            return

        logging.info(f"더미 NIfTI 파일 생성 중: {file_path}")
        
        if is_label:
            # 레이블 파일: 0, 1, 2 (배경, 췌장, 종양) 값을 가지는 정수형 데이터
            array_data = np.random.randint(0, 3, size=IMAGE_SHAPE, dtype=np.uint8)
        else:
            # 이미지 파일: 0-255 사이의 값을 가지는 float 데이터
            array_data = np.random.uniform(0, 255, size=IMAGE_SHAPE).astype(np.float32)

        sitk_image = sitk.GetImageFromArray(array_data)
        sitk_image.SetSpacing(IMAGE_SPACING)
        
        sitk.WriteImage(sitk_image, file_path)

    except Exception as e:
        logging.error(f"NIfTI 파일 생성 실패: {file_path}, 오류: {e}")
        raise

def create_manifest(manifest_path: str, file_list: list):
    """데이터 목록(manifest) JSON 파일을 생성합니다."""
    try:
        logging.info(f"Manifest 파일 생성 중: {manifest_path}")
        
        manifest_data = []
        for img_path, lbl_path in file_list:
            # 스크립트 실행 위치에 관계없이 일관된 상대 경로를 사용하도록 수정
            rel_img_path = os.path.relpath(img_path, DATA_DIR)
            rel_lbl_path = os.path.relpath(lbl_path, DATA_DIR)
            manifest_data.append({
                "image": rel_img_path.replace(os.sep, '/'),
                "label": rel_lbl_path.replace(os.sep, '/')
            })

        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=4)

    except Exception as e:
        logging.error(f"Manifest 파일 생성 실패: {manifest_path}, 오류: {e}")
        raise

def main():
    """메인 실행 함수"""
    logging.info("--- 더미 데이터 생성 스크립트 시작 ---")

    # 1. 필요한 디렉토리 생성
    logging.info("필요한 디렉토리를 확인하고 생성합니다...")
    os.makedirs(NIFTI_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    os.makedirs(MANIFESTS_DIR, exist_ok=True)
    logging.info(f"데이터 디렉토리: {DATA_DIR}")

    # 2. 더미 NIfTI 파일 (이미지 및 레이블) 생성
    logging.info("더미 NIfTI 이미지 및 레이블 파일을 생성합니다...")
    
    train_files = []
    val_files = []
    test_files = []

    # 학습 데이터
    for i in range(NUM_TRAIN):
        img_path = os.path.join(NIFTI_DIR, f'train_image_{i+1:03d}.nii.gz')
        lbl_path = os.path.join(LABELS_DIR, f'train_label_{i+1:03d}.nii.gz')
        create_dummy_nifti(img_path, is_label=False)
        create_dummy_nifti(lbl_path, is_label=True)
        train_files.append((img_path, lbl_path))

    # 검증 데이터
    for i in range(NUM_VAL):
        img_path = os.path.join(NIFTI_DIR, f'val_image_{i+1:03d}.nii.gz')
        lbl_path = os.path.join(LABELS_DIR, f'val_label_{i+1:03d}.nii.gz')
        create_dummy_nifti(img_path, is_label=False)
        create_dummy_nifti(lbl_path, is_label=True)
        val_files.append((img_path, lbl_path))

    # 테스트 데이터
    for i in range(NUM_TEST):
        img_path = os.path.join(NIFTI_DIR, f'test_image_{i+1:03d}.nii.gz')
        lbl_path = os.path.join(LABELS_DIR, f'test_label_{i+1:03d}.nii.gz')
        create_dummy_nifti(img_path, is_label=False)
        create_dummy_nifti(lbl_path, is_label=True)
        test_files.append((img_path, lbl_path))

    # 3. Manifest 파일 생성
    logging.info("Manifest JSON 파일을 생성합니다...")
    create_manifest(os.path.join(MANIFESTS_DIR, 'pancreas_ct_train.json'), train_files)
    create_manifest(os.path.join(MANIFESTS_DIR, 'pancreas_ct_val.json'), val_files)
    create_manifest(os.path.join(MANIFESTS_DIR, 'pancreas_ct_test.json'), test_files)

    logging.info("--- ✅ 더미 데이터 생성 완료! ---")
    print("\n이제 다음 명령어로 학습 테스트를 실행할 수 있습니다:")
    print("python -m pancreas_cancer_diagnosis.segmentation.training.train --model-name unet --epochs 1 --batch-size 1")


if __name__ == '__main__':
    main()
