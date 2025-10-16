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

### 1. 세그멘테이션 모델 학습 (독립 실행)

5개의 세그멘테이션 모델을 각각 독립적으로 학습:

```bash
# UNet 학습 (모델 1/5)
python -m pancreas_cancer_diagnosis.segmentation.training.train \
    --config pancreas_cancer_diagnosis/configs/segmentation/unet_example.yaml \
    --model unet \
    --gpus 4 \
    --seed 42

# ResUNet 학습 (모델 2/5)
python -m pancreas_cancer_diagnosis.segmentation.training.train \
    --config pancreas_cancer_diagnosis/configs/segmentation/resunet_example.yaml \
    --model resunet \
    --gpus 4 \
    --seed 42

# 마찬가지로 vnet, attention_unet, c2fnas도 학습...
```

각 모델은 독립적으로 학습되며 체크포인트를 `outputs/segmentation/<model_name>/`에 저장합니다.

### 2. 세그멘테이션 출력 생성

학습 후, 분류를 위한 세그멘테이션 마스크를 생성:

```bash
python -m pancreas_cancer_diagnosis.segmentation.inference.inference \
    --model unet \
    --checkpoint outputs/segmentation/unet/checkpoints/best.pth \
    --input_dir data/ct_images \
    --output_dir data/segmentations/unet
```

5개 모델 모두에 대해 반복하여 5세트의 세그멘테이션 마스크를 생성합니다.

### 3. 분류 모델 학습 (독립 실행)

세그멘테이션 출력으로 분류 모델 학습:

```bash
# UNet 세그멘테이션으로 분류기 1 학습
python -m pancreas_cancer_diagnosis.classification.training.train \
    --config pancreas_cancer_diagnosis/configs/classification/resnet3d_example.yaml \
    --model resnet3d \
    --gpus 1 \
    --use_segmentation \
    --seed 42

# ResUNet 세그멘테이션으로 분류기 2 학습
# ... 5개 모두 반복
```

### 4. End-to-End 추론

새로운 CT 스캔에 대해 전체 파이프라인 실행:

```bash
python -m pancreas_cancer_diagnosis.pipeline.inference \
    --config pancreas_cancer_diagnosis/configs/pipeline/inference_example.yaml \
    --input data/test_patient_001.nii.gz \
    --output results/patient_001 \
    --save_segmentations
```

**출력 예시:**
```json
{
  "input_file": "data/test_patient_001.nii.gz",
  "prediction": 1,
  "diagnosis": "암",
  "probabilities": {
    "정상": 0.23,
    "암": 0.77
  },
  "uncertainty": {
    "정상": 0.05,
    "암": 0.05
  }
}
```

## 📊 워크플로우

### 전체 학습 파이프라인

1. **데이터 준비**
   - CT 스캔을 NIfTI 포맷(`.nii.gz`)으로 변환
   - train/val/test 분할이 포함된 `datalist.json` 생성

2. **5개 세그멘테이션 모델 학습**
   ```bash
   for model in unet resunet vnet attention_unet c2fnas; do
       python -m pancreas_cancer_diagnosis.segmentation.training.train \
           --config configs/segmentation/${model}.yaml \
           --model ${model} \
           --seed 42
   done
   ```

3. **세그멘테이션 출력 생성**
   ```bash
   # 데이터셋에 대해 5개 모델 모두로 추론 실행
   ```

4. **5개 분류 모델 학습**
   ```bash
   # 각 세그멘테이션 출력마다 하나의 분류기 학습
   ```

5. **End-to-End 추론 실행**
   ```bash
   # 최종 진단을 위한 전체 파이프라인 사용
   ```

## 🔧 설정

모든 설정은 YAML 형식 사용:

### 세그멘테이션 설정
```yaml
model:
  in_channels: 1
  num_classes: 3

data:
  data_root: "/path/to/data"
  batch_size: 2

training:
  learning_rate: 0.001
  max_epochs: 100
```

### 파이프라인 설정
```yaml
segmentation_models:
  - type: "unet"
    checkpoint: "path/to/unet.pth"
  # ... 4개 더

classification_models:
  - type: "resnet3d"
    checkpoint: "path/to/classifier.pth"
  # ... 4개 더

ensemble_method: "weighted"
ensemble_weights: [0.25, 0.20, 0.20, 0.20, 0.15]
```

## 🎯 재현성

프레임워크에는 재현성을 보장하는 도구가 포함되어 있습니다:

```python
from pancreas_cancer_diagnosis.pipeline.orchestrator import ReproducibilityManager

# 실험 설정 추적
repro_manager = ReproducibilityManager(experiment_dir="experiments/run_001")

# 각 모델 설정 로깅
repro_manager.log_segmentation_config(model_idx=0, config=model_config)
repro_manager.log_results(split="test", metrics=test_metrics)

# 모든 정보 저장
repro_manager.save_experiment_info()
```

## 🔑 핵심 설계 원칙

### 1. 모듈화
각 모듈(세그멘테이션, 분류)은 완전히 독립적:
- 자체 모델, 학습 스크립트, 추론 스크립트 보유
- 개별적으로 개발 및 테스트 가능
- 새로운 모델 추가 용이

### 2. 유연성
- **모델 쉽게 교체**: 모든 모델이 베이스 클래스 상속
- **다양한 앙상블 전략**: Average, weighted, voting, stacking
- **설정 가능**: YAML 설정으로 모든 것 제어

### 3. 유지보수성
- **명확한 인터페이스**: 베이스 클래스가 계약 정의
- **관심사 분리**: 데이터, 모델, 학습, 추론이 분리됨
- **타입 힌트와 문서화**: 이해하고 확장하기 쉬움

### 4. 독립 실행
각 컴포넌트를 독립적으로 실행 가능:
- 분류 없이 세그멘테이션만 학습
- 전체 파이프라인 없이 분류만 학습
- 모델의 어떤 조합이든 사용 가능

## 🔨 프레임워크 확장

### 새로운 세그멘테이션 모델 추가

1. `BaseSegmentationModel`을 상속하는 새 모델 생성:
```python
# pancreas_cancer_diagnosis/segmentation/models/my_model.py
from .base import BaseSegmentationModel

class MyNewModel(BaseSegmentationModel):
    def __init__(self, in_channels=1, num_classes=3, **kwargs):
        super().__init__(in_channels, num_classes, **kwargs)
        # 여기에 아키텍처 구현

    def forward(self, x):
        # 여기에 forward pass 구현
        return output
```

2. `__init__.py`에 등록
3. 설정 템플릿 추가
4. 독립적으로 학습!

### 새로운 앙상블 전략 추가

`EnsembleClassifier` 확장:
```python
def _my_ensemble_method(self, predictions):
    # 커스텀 앙상블 로직
    return combined_predictions
```

## 📝 인용

연구에서 이 프레임워크를 사용하시면 다음과 같이 인용해주세요:

```bibtex
@software{pancreas_cancer_diagnosis,
  title = {췌장암 진단: End-to-End 파이프라인},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/end_to_end_workflow}
}
```

## 📄 라이선스

MIT License

## 🤝 기여

기여를 환영합니다! 모듈화된 설계로 다음과 같은 작업이 쉽습니다:
- 새로운 모델 추가
- 기존 아키텍처 개선
- 새로운 앙상블 전략 추가
- 데이터 로더 개선

이슈나 풀 리퀘스트를 열어주세요.

## 📧 연락처

질문이나 문제가 있으시면 GitHub 이슈를 열거나 [your-email@example.com]으로 연락주세요.
