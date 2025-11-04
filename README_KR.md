# 췌장암 조기 진단 AI - Stage 1 Detection System

**다른 목적으로 찍은 CT에서 췌장암을 조기에 발견하는 AI 시스템**

정상 췌장의 미세한 변화를 감지하여 증상이 나타나기 전, Stage 1-2에서 췌장암을 발견하는 것을 목표로 합니다.

---

## 🏥 임상적 배경 (Clinical Background)

### 왜 Stage 1 조기 발견이 중요한가?

췌장암은 진단 시점의 병기(Stage)에 따라 생존율이 극적으로 달라집니다:

| 병기 | 5년 생존율 | CT 발견 시 |
|------|-----------|-----------|
| **Stage 1** (T1N0M0, <2cm) | **~80%** | 거의 불가능 ❌ |
| **Stage 2** (T2-3, 2-4cm) | ~30% | 매우 어려움 |
| **Stage 3-4** (국소 진행/전이) | **<5%** | **대부분 여기서 발견** ✓ |

**현실:** CT에서 췌장암이 발견될 때는 대부분 이미 Stage 3-4입니다.

### CT에서 조기 발견이 어려운 이유

1. **대비도(Contrast) 문제**
   - 1-2cm 이하의 작은 종양은 정상 췌장 조직과 밀도 차이가 거의 없음
   - 일반 복부 CT는 췌장에 최적화된 타이밍이 아님

2. **증상의 부재**
   - Stage 1-2: 증상이 거의 없음 → 검사 자체를 안 함
   - **황달 증상**: 담관이 막혀야 나타남 → 이미 Stage 3-4

3. **검진의 한계**
   - 건강검진: 주로 복부 초음파 (췌장 관찰률 50-70%)
   - 건강검진에서 CT를 찍는 경우: **<5%**
   - CT는 주로 "증상이 있어서" 찍힌다 → 그때는 이미 늦음

### 실제 CT가 찍히는 경우

CT는 건강검진보다는 다음 상황에서 주로 찍힙니다:

1. **증상으로 인한 외래/응급실 방문**
   - 복통, 소화불량, 체중 감소, 황달
   - 다른 장기(간, 담낭, 위장) 평가 목적
   - 췌장은 부수적으로만 관찰됨

2. **타 질환 정기 추적 관찰**
   - 간질환, 신장질환, 대장암 수술 후 추적
   - 매년 정기 CT를 찍지만 췌장은 세밀히 안 봄

3. **외상/수술 전 평가**
   - 췌장은 주 관심사가 아님

**💡 핵심 인사이트:** 한국에서만 연간 **500만+ 복부 CT**가 찍히고 있지만, 대부분 췌장은 "대충" 봅니다. 이것이 바로 AI의 기회입니다.

---

## 🎯 프로젝트의 목표와 가치

### Opportunistic Pancreatic Cancer Screening

**"다른 목적으로 찍은 CT에서 췌장도 자동으로 정밀 분석"**

#### 시나리오 A: 증상 평가 중 우연한 발견
```
환자: "소화가 안 되고 배가 더부룩해요"
의사: 복부 CT 처방 (위/담낭 평가 목적)
방사선과: "특이 소견 없음" (췌장은 간단히만 봄)

→ AI 적용 ⭐
   "췌장 body 부위에 subtle texture heterogeneity 감지"
   "Confidence: 68%, 췌장 전용 CT 또는 MRI 권장"

→ 추가 정밀 검사 → Stage 1 (1.5cm) 발견
→ 수술 가능, 완치 기회
```

#### 시나리오 B: 정기 추적 관찰 중 조기 발견
```
환자: 간경화 추적 관찰 (6개월마다 CT)

→ AI의 Longitudinal Analysis ⭐
   "3회 연속 CT 비교 분석"
   "췌장 두부(head) 영역에서 progressive texture change"
   "크기 변화는 없지만 attenuation pattern 변화 감지"

→ Dedicated pancreas protocol CT
→ Stage 1 발견 (증상 나타나기 전)
```

#### 시나리오 C: 대규모 Retrospective Screening
```
병원 PACS에 저장된 과거 CT 스캔들
→ AI가 자동으로 재분석 (Batch Processing)
→ 놓쳤던 의심 소견 발견
→ 해당 환자에게 추적 검사 권유
```

### AI의 핵심 역할

1. **Subtle Feature Detection**
   - 사람 눈으로는 구분하기 어려운 미세한 texture 변화
   - 췌장 경계의 irregularity
   - Parenchymal attenuation의 heterogeneity
   - 췌관의 focal dilation (<3mm)

2. **Longitudinal Monitoring**
   - 동일 환자의 과거 CT와 자동 비교
   - 매우 느린 growth rate도 감지
   - 정상 변이(normal variation)와 pathologic change 구분

3. **Multi-hospital Domain Adaptation**
   - 병원마다 다른 CT 장비, 프로토콜
   - 각 기관의 정상 분포를 학습
   - Domain shift 극복

4. **Uncertainty Quantification**
   - Stage 1은 false positive가 높을 수밖에 없음
   - Confidence score와 함께 추적 관찰 권장 리포트
   - "즉시 추가 검사" vs "3개월 후 재검" 구분

---

## 📌 주요 특징

- **Opportunistic Screening**: 다른 목적으로 찍은 CT에서 췌장 자동 분석
- **Anomaly Detection**: 정상 췌장 학습 후 미세한 이상 패턴 감지
- **Longitudinal Analysis**: 시계열 CT 비교를 통한 progressive change 탐지
- **Multi-hospital Adaptation**: 병원/장비 간 차이를 극복하는 domain adaptation
- **Uncertainty Quantification**: 신뢰도 기반 추적 관찰 또는 즉시 정밀 검사 권장
- **완전한 모듈화**: 세그멘테이션, 분류, 이상 탐지 모듈이 독립적으로 실행 가능

---

## 🔬 CT Imaging 최적화 및 기술적 접근

### Pancreas-Dedicated CT Protocol (이상적인 경우)

Stage 1 췌장암을 보려면 일반 복부 CT보다 더 정밀한 프로토콜이 필요합니다:

#### 1. Multi-phase Contrast Enhancement
```
Late Arterial Phase (Pancreatic Parenchymal Phase)
  - Timing: 40-50초
  - 췌장 실질이 가장 잘 보이는 시점
  - 작은 종양과 정상 조직의 대비 최대화

Portal Venous Phase
  - Timing: 70-80초
  - 주변 혈관 평가
  - 혈관 침범 여부 확인

Delayed Phase (Optional)
  - Timing: 3-5분
  - 일부 hypovascular tumor 감지
```

#### 2. Thin-slice Acquisition
```
Slice Thickness: 0.5-1mm (일반 CT: 3-5mm)
→ Partial volume effect 감소
→ 작은 병변 감지율 향상
→ 3D reconstruction 품질 향상
```

#### 3. High Resolution Settings
```
Matrix: 512x512 이상
Field of View: 췌장에 집중
Reconstruction: 여러 알고리즘 조합
```

### 현실: 일반 복부 CT에서 작동해야 함

**하지만 대부분의 CT는 이렇게 찍히지 않습니다:**
- Single phase 또는 간단한 dual-phase
- 5mm slice thickness
- 췌장이 주 목적이 아님

**따라서 우리 AI는:**
- ✅ **최적이 아닌 CT에서도 작동**해야 합니다
- ✅ **다양한 프로토콜에 robust**해야 합니다
- ✅ **병원마다 다른 장비/설정을 학습**해야 합니다

---

## 🧠 기술적 접근법: Anomaly Detection

### 왜 Anomaly Detection인가?

#### 문제: 암 데이터 부족
```
공개 데이터셋 (NIH Pancreas-CT):
  - 정상 췌장: 82례 ✓
  - 췌장암: 0례 ❌

실제 임상 데이터:
  - 정상/양성 질환: 수십만 건
  - Stage 1-2 췌장암: 수백 건 (매우 희귀)
```

#### 해결책: 정상을 완벽히 학습
```
"정상이 어떤 것인지 완벽히 학습하면,
 정상이 아닌 것(anomaly)을 찾을 수 있다"
```

### 핵심 아이디어

#### 1. U-Net 기반 Autoencoder
```python
# 정상 췌장만으로 학습
Input: 정상 췌장 CT
→ Encoder: 특징 압축
→ Decoder: 원본 복원
Output: 복원된 CT

Loss = MSE(Input, Output)
```

**정상 데이터 학습 후:**
- 정상 pancreas → 완벽하게 복원 (low error)
- 암이 있는 pancreas → 복원 실패 (high error)
- **High error region = Anomaly = 의심 부위**

#### 2. Weighted Reconstruction Loss
```python
# 췌장 영역에 더 높은 가중치
Loss = weighted_MSE(Input, Output, pancreas_mask)

pancreas 영역: weight = 10.0
background: weight = 1.0
```

**이유:**
- 작은 종양(1cm 미만)도 놓치지 않기 위해
- 췌장 내부의 subtle change에 집중
- Background noise는 무시

#### 3. Multi-scale Feature Analysis
```
여러 해상도에서 동시에 분석:
- High resolution: 작은 종양 (<1cm)
- Medium resolution: texture pattern
- Low resolution: 전체적인 형태 변화
```

#### 4. Temporal Consistency (향후 계획)
```
동일 환자의 과거 CT와 비교:
- t0: 정상 (baseline)
- t1: 미세한 변화 (AI 감지)
- t2: 명확한 변화 (확진)

→ Progressive change pattern 학습
→ False positive 감소
```

### Expected Output Example

```json
{
  "patient_id": "P001234",
  "scan_date": "2025-10-15",
  "anomaly_detected": true,
  "anomaly_score": 0.73,
  "recommendation": "췌장 전용 CT 또는 MRI 권장",
  "confidence": "medium-high",
  "region_of_interest": {
    "location": "pancreatic body",
    "size_estimate": "8-12mm",
    "reconstruction_error": 0.089
  },
  "follow_up": {
    "urgency": "non-urgent",
    "suggested_interval": "3 months",
    "reason": "subtle texture heterogeneity without definite mass"
  }
}
```

---

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
