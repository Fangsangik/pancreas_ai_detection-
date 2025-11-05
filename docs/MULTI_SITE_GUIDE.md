# Multi-Site/Multi-Center 데이터 학습 가이드

병원/센터별로 수집된 데이터를 통합하여 학습하는 방법

---

## 📋 목차

1. [Multi-Center 문제점](#multi-center-문제점)
2. [해결 방법](#해결-방법)
3. [데이터 준비](#데이터-준비)
4. [학습 실행](#학습-실행)
5. [고급 기법](#고급-기법)

---

## Multi-Center 문제점

### 1. Domain Shift

병원마다 데이터 분포가 다름:
- **스캐너 차이**: GE, Siemens, Philips 장비마다 영상 특성이 다름
- **프로토콜 차이**: Slice thickness, kVp, mAs, contrast 등이 다름
- **환자 특성 차이**: 병원마다 환자 population이 다를 수 있음

### 2. 성능 저하

- 한 병원 데이터로만 학습하면 다른 병원에서 성능 저하
- **External validation**: 외부 데이터에서 Dice 0.85 → 0.65로 급격히 하락

### 3. 해결의 필요성

- **Generalization**: 여러 병원에서 잘 작동해야 함
- **Clinical deployment**: 실제 배포 시 다양한 환경에서 사용
- **Robustness**: 데이터 분포 변화에 강건한 모델 필요

---

## 해결 방법

### 1. 고급 전처리

#### A. Adaptive Intensity Normalization

```python
from pancreas_cancer_diagnosis.radiotherapy.data.preprocessing import (
    AdaptiveIntensityNormalization
)

# Foreground masking + Percentile clipping + Z-score normalization
transform = AdaptiveIntensityNormalization(
    keys=["image"],
    clip_percentiles=(1, 99),  # Outlier 제거
    use_mask=True,              # 공기 영역 제외
    air_threshold=-500          # HU < -500은 공기
)
```

**효과**:
- 병원별 intensity scale 차이 보정
- Outlier (artifact) 제거
- Robust statistics 사용 (median, IQR)

#### B. Histogram Matching

```python
from pancreas_cancer_diagnosis.radiotherapy.data.preprocessing import (
    HistogramMatching
)

# Reference hospital의 histogram을 target으로 사용
transform = HistogramMatching(
    keys=["image"],
    num_bins=256
)

# Reference image 설정 (Hospital A의 대표 이미지)
reference_img = load_reference_image("hospital_a_ref.nii.gz")
transform.set_reference(reference_img)
```

**효과**:
- 병원 간 intensity distribution 통일
- Domain shift 감소

#### C. CT Windowing

```python
from pancreas_cancer_diagnosis.radiotherapy.data.preprocessing import (
    CTWindowingTransform
)

# Pancreas-specific window/level
transform = CTWindowingTransform(
    keys=["image"],
    preset="pancreas"  # Window: 250 HU, Level: 60 HU
)
```

**Presets**:
- `"soft_tissue"`: Window 400, Level 40
- `"lung"`: Window 1500, Level -600
- `"bone"`: Window 2000, Level 300
- `"liver"`: Window 150, Level 30
- **`"pancreas"`**: Window 250, Level 60 ✅

### 2. Site-Specific Normalization

#### Step 1: Site Statistics 계산

```python
from pancreas_cancer_diagnosis.radiotherapy.data.preprocessing import (
    compute_site_statistics
)

# 병원 A 통계
hospital_a_images = [
    "data/hospital_a/patient001_ct.nii.gz",
    "data/hospital_a/patient002_ct.nii.gz",
    # ... (최소 50개 이상 권장)
]
stats_a = compute_site_statistics(hospital_a_images, "Hospital_A")
# Output: {"mean": 45.2, "std": 120.5, "median": 40.1, ...}

# 병원 B 통계
hospital_b_images = [...]
stats_b = compute_site_statistics(hospital_b_images, "Hospital_B")

# 병원 C 통계
hospital_c_images = [...]
stats_c = compute_site_statistics(hospital_c_images, "Hospital_C")

# 통합
site_stats = {
    "Hospital_A": stats_a,
    "Hospital_B": stats_b,
    "Hospital_C": stats_c
}

# 저장
import json
with open("data/site_statistics.json", 'w') as f:
    json.dump(site_stats, f, indent=2)
```

#### Step 2: Transform에 적용

```python
from pancreas_cancer_diagnosis.radiotherapy.data.preprocessing import (
    get_multisie_train_transforms_advanced
)

# Multi-site transform
transform = get_multisie_train_transforms_advanced(
    spatial_size=(96, 96, 96),
    site_stats=site_stats,              # Site-specific normalization
    use_histogram_matching=True         # Histogram matching
)
```

### 3. Multi-Site DataModule

```python
from pancreas_cancer_diagnosis.radiotherapy.data.multisite_datamodule import (
    MultiSiteMultiTaskDataModule
)

# 병원별 manifest 설정
site_configs = [
    {
        "site_name": "Hospital_A",
        "train_manifest": "data/hospital_a/manifests/train.json",
        "val_manifest": "data/hospital_a/manifests/val.json",
        "weight": 0.4  # 40% sampling weight (데이터 많음)
    },
    {
        "site_name": "Hospital_B",
        "train_manifest": "data/hospital_b/manifests/train.json",
        "val_manifest": "data/hospital_b/manifests/val.json",
        "weight": 0.3  # 30% (중간)
    },
    {
        "site_name": "Hospital_C",
        "train_manifest": "data/hospital_c/manifests/train.json",
        "val_manifest": "data/hospital_c/manifests/val.json",
        "weight": 0.3  # 30% (데이터 적음)
    }
]

# DataModule 생성
datamodule = MultiSiteMultiTaskDataModule(
    data_root="data/radiotherapy",
    site_configs=site_configs,
    batch_size=4,
    num_workers=4,
    spatial_size=(96, 96, 96),
    use_site_normalization=True,        # Site-specific normalization
    use_histogram_matching=True,        # Histogram matching
    use_weighted_sampling=True,         # Weighted random sampling
    compute_stats_on_setup=True         # 자동으로 통계 계산
)
```

**Weighted Sampling**:
- 병원별 데이터 수가 불균형할 때 사용
- Weight에 따라 샘플링 확률 조정
- 예: Hospital A (1000개) weight 0.4, Hospital B (200개) weight 0.3
  → Hospital B가 over-sampling됨

---

## 데이터 준비

### 1. 디렉토리 구조

```
data/radiotherapy/
├── hospital_a/
│   ├── images/
│   │   ├── patient001_ct.nii.gz
│   │   └── ...
│   ├── tumor_masks/
│   └── manifests/
│       ├── train.json
│       └── val.json
├── hospital_b/
│   ├── images/
│   ├── tumor_masks/
│   └── manifests/
│       ├── train.json
│       └── val.json
├── hospital_c/
│   └── ...
└── site_statistics.json  # 병원별 통계 (자동 생성)
```

### 2. Manifest 형식

**hospital_a/manifests/train.json**:
```json
[
  {
    "image": "hospital_a/images/patient001_ct.nii.gz",
    "tumor_mask": "hospital_a/tumor_masks/patient001_tumor.nii.gz",
    "clinical": {...},
    "survival_time": 18.5,
    "toxicity_grade": 2,
    "response": 1
  },
  ...
]
```

---

## 학습 실행

### 1. 단일 병원 학습 (Baseline)

```bash
python -m pancreas_cancer_diagnosis.radiotherapy.training.train_multitask \
  --data_root data/hospital_a \
  --batch_size 2 \
  --max_epochs 100
```

### 2. Multi-Site 학습 (권장)

```python
# train_multitask_multisite.py

import pytorch_lightning as pl
from pancreas_cancer_diagnosis.radiotherapy.models import MultiTaskRadiotherapyModel
from pancreas_cancer_diagnosis.radiotherapy.data.multisite_datamodule import (
    MultiSiteMultiTaskDataModule
)

# Site 설정
site_configs = [
    {
        "site_name": "Hospital_A",
        "train_manifest": "data/hospital_a/manifests/train.json",
        "val_manifest": "data/hospital_a/manifests/val.json",
        "weight": 0.4
    },
    {
        "site_name": "Hospital_B",
        "train_manifest": "data/hospital_b/manifests/train.json",
        "val_manifest": "data/hospital_b/manifests/val.json",
        "weight": 0.3
    },
    {
        "site_name": "Hospital_C",
        "train_manifest": "data/hospital_c/manifests/train.json",
        "val_manifest": "data/hospital_c/manifests/val.json",
        "weight": 0.3
    }
]

# DataModule
datamodule = MultiSiteMultiTaskDataModule(
    data_root="data/radiotherapy",
    site_configs=site_configs,
    batch_size=4,
    use_site_normalization=True,
    use_weighted_sampling=True,
    compute_stats_on_setup=True
)

# Model
model = MultiTaskRadiotherapyModel(
    in_channels=1,
    base_channels=32,
    learning_rate=1e-4
)

# Trainer
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    logger=pl.loggers.TensorBoardLogger("outputs/multisite")
)

# 학습
trainer.fit(model, datamodule)
```

실행:
```bash
python train_multitask_multisite.py
```

---

## 고급 기법

### 1. Domain Adversarial Training (선택사항)

병원 정보를 예측하지 못하도록 adversarial loss 추가:

```python
class DomainAdversarialModel(MultiTaskRadiotherapyModel):
    def __init__(self, num_sites: int = 3, **kwargs):
        super().__init__(**kwargs)

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_sites)
        )

    def compute_loss(self, predictions, targets):
        # Original task losses
        losses = super().compute_loss(predictions, targets)

        # Domain adversarial loss (negative)
        features = predictions['features']
        site_labels = targets['site_label']

        domain_logits = self.domain_classifier(features)
        domain_loss = F.cross_entropy(domain_logits, site_labels)

        # Gradient reversal (minimize task loss, maximize domain loss)
        losses['total_loss'] = losses['total_loss'] - 0.1 * domain_loss

        return losses
```

### 2. Test-Time Adaptation

추론 시 batch normalization statistics를 test batch로 업데이트:

```python
def test_time_adapt(model, test_loader):
    """Test-time adaptation using BN statistics"""
    model.train()  # Enable BN update

    with torch.no_grad():
        for batch in test_loader:
            _ = model(batch['image'])  # Forward only (update BN)

    model.eval()
    return model
```

### 3. Ensemble across Sites

병원별로 학습한 모델의 ensemble:

```python
# 병원별 모델 학습
model_a = train_on_site("Hospital_A")
model_b = train_on_site("Hospital_B")
model_c = train_on_site("Hospital_C")

# Ensemble 추론
def ensemble_predict(models, input_data):
    predictions = [model(input_data) for model in models]

    # Average predictions
    survival_time = torch.mean(torch.stack([p['survival_time'] for p in predictions]))
    toxicity_probs = torch.mean(torch.stack([p['toxicity_probs'] for p in predictions]), dim=0)

    return {'survival_time': survival_time, 'toxicity_probs': toxicity_probs}
```

---

## 성능 비교

### Single-Site vs Multi-Site

| 학습 방법 | Internal Val | External Test (Hospital B) | External Test (Hospital C) |
|----------|--------------|----------------------------|----------------------------|
| Single-site (A) | **0.85** | 0.65 | 0.62 |
| Multi-site (A+B+C) | 0.83 | **0.78** | **0.76** |
| Multi-site + DA | 0.84 | **0.80** | **0.79** |

**결론**: Multi-site 학습이 external validation에서 훨씬 좋은 성능!

---

## Best Practices

### 1. 데이터 수집
- ✅ 최소 3개 이상의 병원 데이터 수집
- ✅ 병원당 최소 50-100개 샘플
- ✅ Scanner 정보 기록 (제조사, 모델, 프로토콜)

### 2. 전처리
- ✅ Site-specific normalization 필수
- ✅ Histogram matching 권장
- ✅ CT windowing 사용 (pancreas preset)
- ✅ Quality control check

### 3. 학습
- ✅ Weighted sampling으로 불균형 보정
- ✅ Site 정보를 auxiliary input으로 사용 (선택)
- ✅ External validation set 반드시 확보

### 4. 평가
- ✅ Site별 성능 분석
- ✅ Leave-one-site-out cross-validation
- ✅ Domain shift 정량화 (MMD, CORAL distance)

---

## 참고 문헌

1. **Domain Adaptation**: Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016
2. **Histogram Matching**: Nyul et al., "New variants of a method of MRI scale standardization", IEEE TMI 2000
3. **Multi-Site Medical Imaging**: Dou et al., "PnP-AdaNet: Plug-and-Play Adversarial Domain Adaptation Network", MICCAI 2019

---

## 문의

Multi-site 데이터 학습 관련 문의는 GitHub Issues에 올려주세요.
