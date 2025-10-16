# 췌장암 진단 - End-to-End 파이프라인

**5개의 독립적인 세그멘테이션 CNN**과 **앙상블 분류 CNN**을 사용한 모듈화되고 유연하며 재현 가능한 췌장암 진단 프레임워크입니다.

## 📌 주요 특징

- **완전한 모듈화**: 각 모듈(세그멘테이션, 분류, 파이프라인)이 독립적으로 실행 가능
- **높은 유연성**: 모델 교체, 새로운 아키텍처 추가, 워크플로우 수정이 쉬움
- **쉬운 유지보수**: 명확한 관심사 분리와 잘 정의된 인터페이스
- **재현성 보장**: 설정 추적 및 재현 가능한 결과를 위한 내장 도구
- **독립 실행**: 각 컴포넌트를 개별적으로 학습하고 테스트하거나 전체 파이프라인 사용 가능

## 🏗️ 프로젝트 구조

```
pancreas_cancer_diagnosis/
├── segmentation/           # 5개 세그멘테이션 CNN (독립 모듈)
│   ├── models/            # UNet, ResUNet, VNet, AttentionUNet, C2FNAS
│   ├── training/          # 독립 실행 학습 스크립트
│   └── inference/         # 독립 실행 추론 스크립트
├── classification/         # 분류 CNN (독립 모듈)
│   ├── models/            # ResNet3D, DenseNet3D, Ensemble
│   ├── training/          # 독립 실행 학습 스크립트
│   └── inference/         # 독립 실행 추론 스크립트
├── pipeline/              # End-to-end 오케스트레이터
│   ├── orchestrator.py    # 메인 파이프라인 컨트롤러
│   └── inference.py       # End-to-end 추론 스크립트
├── data/                  # 공유 데이터 로더
│   ├── dataset.py         # PyTorch 데이터셋
│   └── datamodule.py      # Lightning 데이터 모듈
├── utils/                 # 유틸리티
└── configs/               # 설정 파일 템플릿
    ├── segmentation/      # 각 세그멘테이션 모델 설정
    ├── classification/    # 분류 모델 설정
    └── pipeline/          # End-to-end 파이프라인 설정
```

## 🚀 설치

```bash
# 저장소 클론
git clone <your-repo-url>
cd end_to_end_workflow

# 패키지 설치
pip install -r requirements.txt

# 또는 개발 모드로 설치
pip install -e .
```

## 💡 사용법

(이하 내용은 기존 지도 학습 파이프라인에 대한 설명입니다.)

---

## 🌟 프로젝트 개발 및 수정 기록 (2025-10-15)

### 이상 탐지(Anomaly Detection) 접근법 도입

사용 가능한 공공 데이터셋(NIH Pancreas-CT)에 췌장암 케이스가 포함되어 있지 않고, 정상 췌장 데이터 및 세그멘테이션 마스크만 사용 가능하다는 사실을 확인했습니다. 이에 따라, 기존의 지도 학습(Supervised Learning) 기반의 암 분류 프로젝트 목표를 **비지도 학습(Unsupervised Learning) 기반의 이상 탐지**로 전환했습니다.

새로운 목표는 정상 췌장의 형태와 구조를 완벽하게 학습하는 '복원 모델(Reconstruction Model)'을 만들고, 이 모델이 제대로 복원하지 못하는 영역을 '이상 부위(Anomaly)'로 탐지하는 것입니다.

### 주요 변경 및 추가 사항

1.  **신규 `anomaly_detection` 모듈 추가**
    -   기존의 지도 학습 파이프라인(`segmentation`, `classification`)은 그대로 보존하면서, 새로운 이상 탐지 파이프라인을 위한 `pancreas_cancer_diagnosis/anomaly_detection` 모듈을 추가했습니다.
    -   핵심 학습 로직은 `train_autoencoder.py`에 구현되었습니다.

2.  **U-Net 기반 오토인코더(Autoencoder) 구현**
    -   기존 `UNet3D` 모델을 복원 모델(Autoencoder)로 활용하는 `LitAutoencoder` 클래스를 구현했습니다.
    -   기존에 비어있던 `UNet3D` 모델의 Encoder, Decoder, forward pass 로직을 완전하게 구현하여 실제 작동하도록 수정했습니다.

3.  **가중치 손실 함수 (Weighted Loss Function) 적용**
    -   작은 종양도 효과적으로 탐지할 수 있도록, 췌장 영역의 복원 오류에 더 큰 가중치를 부여하는 `WeightedMSELoss`를 구현했습니다. 이를 통해 모델이 췌장 영역을 더 세밀하게 학습하도록 유도합니다.

4.  **시각화 콜백 (Visualization Callback) 추가**
    -   학습 중 모델의 성능을 직관적으로 확인할 수 있도록 `matplotlib` 기반의 시각화 콜백을 추가했습니다.
    -   이 콜백은 검증 단계마다 원본 이미지, 모델이 복원한 이미지, 그리고 둘의 차이를 보여주는 오류 맵(Error Map)을 이미지 파일(`outputs/anomaly_detection/visualizations/`)로 저장합니다.

5.  **데이터 파이프라인 디버깅 및 스크립트 추가**
    -   **데이터 변환:** 실제 DICOM 데이터셋을 NIfTI 형식으로 변환하는 파이프라인을 실행하고, 경로 관련 문제를 해결했습니다.
    -   **데이터 목록 생성:** 변환된 데이터를 학습/검증/테스트용으로 나누는 `prepare_pancreas_data.py`의 버그(JSON 직렬화 오류)를 수정했습니다.
    -   **가짜 레이블 생성:** 가중치 손실 함수 테스트를 위해, 실제 췌장 위치 레이블이 없는 현 상황에서 가상의 췌장 영역 레이블을 생성하고, 이를 데이터 목록에 연결하는 `add_dummy_labels.py` 스크립트를 모듈화하여 추가했습니다.

6.  **학습 환경 문제 해결**
    -   **메모리 부족 (Out of Memory):** 3D 데이터의 메모리 사용량 문제를 해결하기 위해, MONAI의 `Resized` Transform을 추가하여 학습 시 이미지 크기를 동적으로 조절하도록 수정했습니다.
    -   **하드웨어 호환성:** Apple Silicon GPU(MPS)에서 `MaxPool3d` 연산이 지원되지 않는 문제를 `PYTORCH_ENABLE_MPS_FALLBACK=1` 환경 변수를 사용하여 해결하고, `Trainer`가 MPS 가속기를 올바르게 인식하도록 코드를 수정했습니다.

이러한 과정을 통해, 현재 프로젝트는 **정상적으로 작동하는 End-to-End 이상 탐지 모델 학습 파이프라인**을 갖추게 되었습니다.
