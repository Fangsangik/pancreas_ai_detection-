#!/bin/bash
# Quick Start Script for TCIA Pancreas CT Data
# 췌장 CT 데이터 학습 빠른 시작 스크립트

set -e  # 오류 발생 시 중단

echo "========================================================================"
echo "🚀 췌장암 진단 모델 학습 빠른 시작"
echo "========================================================================"
echo ""

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 디렉토리 생성
mkdir -p data/{raw_dicom,nifti,labels,manifests}
mkdir -p outputs/{segmentation,seg_predictions,classification}
mkdir -p results

echo "✓ 디렉토리 구조 생성 완료"
echo ""

# Step 1: TCIA 다운로드 확인
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: TCIA 데이터 다운로드"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -d "./data/raw_dicom" ] || [ -z "$(ls -A ./data/raw_dicom)" ]; then
    echo -e "${YELLOW}⚠️  DICOM 데이터가 없습니다${NC}"
    echo ""
    echo "다운로드 방법:"
    echo "1. NBIA Data Retriever 설치"
    echo "   https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images"
    echo ""
    echo "2. 매니페스트 파일 사용:"
    echo "   /Users/hwangsang-ik/Downloads/Pancreas-CT-20200910.tcia"
    echo ""
    echo "3. 다운로드 위치: ./data/raw_dicom/"
    echo ""
    read -p "다운로드 완료 후 Enter를 누르세요..."
else
    echo -e "${GREEN}✓ DICOM 데이터 확인 완료${NC}"
fi
echo ""

# Step 2: DICOM → NIfTI 변환
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: DICOM → NIfTI 변환"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -d "./data/nifti" ] || [ -z "$(ls -A ./data/nifti/*.nii.gz 2>/dev/null)" ]; then
    echo "🔄 DICOM to NIfTI 변환 시작..."
    python scripts/convert_dicom_to_nifti.py \
        --input ./data/raw_dicom \
        --output ./data/nifti
    echo -e "${GREEN}✓ 변환 완료${NC}"
else
    echo -e "${GREEN}✓ NIfTI 파일 이미 존재${NC}"
fi
echo ""

# Step 3: 데이터 검증 및 분할
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: 데이터 검증 및 분할"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -f "./data/manifests/pancreas_ct_train.json" ]; then
    echo "🔍 데이터 검증 및 분할 중..."
    python scripts/prepare_pancreas_data.py \
        --input ./data/nifti \
        --output ./data/manifests \
        --train-ratio 0.7 \
        --val-ratio 0.15 \
        --test-ratio 0.15 \
        --random-seed 42
    echo -e "${GREEN}✓ 데이터 분할 완료${NC}"
else
    echo -e "${GREEN}✓ 매니페스트 파일 이미 존재${NC}"
fi
echo ""

# Step 4: 레이블 확인
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: 세그멘테이션 레이블 확인"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -d "./data/labels" ] || [ -z "$(ls -A ./data/labels/*.nii.gz 2>/dev/null)" ]; then
    echo -e "${RED}❌ 레이블 파일이 없습니다!${NC}"
    echo ""
    echo "레이블 준비 방법:"
    echo "1. NIH Pancreas-CT 데이터셋 레이블 다운로드"
    echo "   https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT"
    echo ""
    echo "2. 3D Slicer로 수동 어노테이션"
    echo "   https://www.slicer.org/"
    echo ""
    echo "3. 레이블 파일을 ./data/labels/ 에 저장"
    echo "   형식: PANCREAS_XXXX_seg.nii.gz"
    echo ""
    read -p "레이블 준비 후 Enter를 누르세요 (또는 Ctrl+C로 종료)..."
else
    label_count=$(ls -1 ./data/labels/*.nii.gz 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ 레이블 파일 확인 완료: ${label_count}개${NC}"
fi
echo ""

# Step 5: 5개 세그멘테이션 모델 학습 여부 확인
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: 세그멘테이션 모델 학습"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "5개 모델 학습 (예상 시간: 20-40시간):"
echo "  - UNet3D"
echo "  - ResUNet3D"
echo "  - VNet"
echo "  - AttentionUNet3D"
echo "  - C2FNAS"
echo ""

read -p "세그멘테이션 모델 학습을 시작하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    for model in unet resunet vnet attunet c2fnas; do
        echo ""
        echo "🧠 $model 학습 시작..."
        python -m pancreas_cancer_diagnosis.segmentation.training.train \
            --model-name $model \
            --data-root ./data/nifti \
            --label-root ./data/labels \
            --train-list ./data/manifests/pancreas_ct_train.json \
            --val-list ./data/manifests/pancreas_ct_val.json \
            --epochs 100 \
            --batch-size 2 \
            --output-dir ./outputs/segmentation/$model \
            --device cuda || echo -e "${YELLOW}⚠️ $model 학습 실패 (계속 진행)${NC}"
    done
    echo -e "${GREEN}✓ 모든 세그멘테이션 모델 학습 완료${NC}"
else
    echo "⏭️  세그멘테이션 학습 건너뛰기"
fi
echo ""

# 완료 메시지
echo "========================================================================"
echo "🎉 초기 설정 완료!"
echo "========================================================================"
echo ""
echo "다음 단계:"
echo "1. 세그멘테이션 모델 학습 (위에서 건너뛴 경우)"
echo "2. 세그멘테이션 추론으로 마스크 생성"
echo "3. 분류 모델 학습 (Threshold Voting)"
echo "4. End-to-End 평가"
echo ""
echo "상세 가이드: 실전_학습_가이드.md"
echo "========================================================================"
