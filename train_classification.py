#!/usr/bin/env python3
"""
분류 모델 학습 래퍼 스크립트

이 스크립트는 pancreas_cancer_diagnosis.classification.training.train 모듈을
올바르게 실행하기 위한 래퍼입니다.

사용법:
    python train_classification.py --model resnet3d --config configs/classification/resnet3d_example.yaml
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 실제 train 모듈 임포트 및 실행
if __name__ == '__main__':
    from pancreas_cancer_diagnosis.classification.training.train import main
    main()
