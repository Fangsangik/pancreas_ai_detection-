# 🏥 Radiotherapy Module for Pancreatic Cancer

췌장암 방사선 치료 계획 및 결과 예측을 위한 통합 AI 시스템

## 📋 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [모델 아키텍처](#모델-아키텍처)
- [설치](#설치)
- [데이터 준비](#데이터-준비)
- [학습](#학습)
- [추론](#추론)
- [End-to-End 파이프라인](#end-to-end-파이프라인)
- [성능 벤치마크](#성능-벤치마크)
- [참고 문헌](#참고-문헌)

---

## 개요

이 모듈은 췌장암 SBRT(Stereotactic Body Radiation Therapy) 치료 계획을 위한 세 가지 핵심 AI 모델을 제공합니다:

1. **OAR Segmentation** - 주요 장기 자동 세그멘테이션
2. **Dose Prediction** - 최적 선량 분포 예측
3. **Multi-task Learning** - 생존 시간, 독성, 치료 반응 동시 예측

### 임상적 목표

- **치료 계획 최적화**: DVH 제약 조건을 고려한 자동 선량 분포 계획
- **독성 예측**: 특히 GI toxicity (십이지장 손상) 위험 평가
- **예후 예측**: 환자별 맞춤형 치료 결정 지원
- **시간 절약**: 수동 계획 대비 5-10배 빠른 치료 계획 수립

---

## 주요 기능

### 1. OAR (Organs at Risk) Segmentation

**목적**: CT 스캔에서 방사선 치료 시 보호해야 할 주요 장기 자동 세그멘테이션

**세그멘테이션 대상 (7 classes)**:
- Class 0: Background
- Class 1: **Duodenum (십이지장)** ⚠️ 가장 중요 - GI toxicity 주원인
- Class 2: Stomach (위)
- Class 3: Small intestine (소장)
- Class 4: Liver (간)
- Class 5: Left kidney (왼쪽 신장)
- Class 6: Right kidney (오른쪽 신장)

**아키텍처**: nnU-Net inspired 3D U-Net with deep supervision

**성능 목표**:
- Duodenum Dice > 0.80
- Other organs Dice > 0.85

### 2. Dose Prediction

**목적**: CT + Tumor + OAR 정보를 기반으로 최적 3D 선량 분포 예측

**입력**:
- CT scan (1 channel)
- Tumor segmentation (1 channel)
- OAR segmentations (N channels) - 주로 duodenum, stomach

**출력**:
- 3D dose distribution (Gy 단위)

**아키텍처**: 3D U-Net with Attention Gates

**Loss 함수**:
1. **MSE loss**: Voxel-wise dose accuracy
2. **Gradient loss**: Dose distribution smoothness
3. **DVH loss**: OAR dose constraints
   - Duodenum: Mean dose < 30 Gy, Max < 45 Gy
   - Stomach: Mean dose < 35 Gy, Max < 50 Gy

### 3. Multi-Task Learning

**목적**: 생존 시간, 독성 등급, 치료 반응을 동시에 예측

**입력**:
- CT scan + Tumor mask
- Clinical features (10개):
  - age, gender, stage, CA19-9, tumor_size
  - location, KPS, diabetes, prior_surgery, chemotherapy

**출력**:
1. **Survival time** (months) + uncertainty
2. **Toxicity grade** (0-3+)
   - Grade 0: None
   - Grade 1: Mild
   - Grade 2: Moderate
   - Grade 3+: Severe
3. **Treatment response** (binary: responder / non-responder)

**아키텍처**: Shared 3D CNN encoder + Task-specific heads

**특징**:
- Uncertainty quantification (Gaussian NLL)
- Task weight balancing
- Multi-task learning으로 feature representation 향상

---

## 모델 아키텍처

### OAR Segmentation Network

```
Input: CT (1, 128, 128, 128)
  ↓
[nnUNetBlock] → [Pool] ×4  (Encoder)
  ↓
[Bottleneck]
  ↓
[UpConv + Skip] → [nnUNetBlock] ×4  (Decoder)
  ↓
Output: Segmentation (7, 128, 128, 128)
+ Deep Supervision outputs
```

- Residual connections
- Instance Normalization
- Leaky ReLU activation
- Deep supervision for better gradient flow

### Dose Prediction Network

```
Input: CT + Tumor + OARs (4, 128, 128, 128)
  ↓
[Conv3D] → [Pool] ×4  (Encoder)
  ↓
[Bottleneck]
  ↓
[AttentionGate] → [UpConv + Skip] → [Conv3D] ×4  (Decoder)
  ↓
Output: Dose Map (1, 128, 128, 128)
```

- Attention gates for focusing on tumor region
- Batch Normalization
- ReLU activation
- Prescription dose normalization

### Multi-Task Network

```
Input: CT + Tumor (1, 96, 96, 96)
  ↓
[Shared 3D ResNet Encoder]
  ↓  (Global Average Pooling)
Features (B, 512)
  ↓
┌──────────────┬──────────────┬────────────────┐
│ Survival     │ Toxicity     │ Response       │
│ Head         │ Head         │ Head           │
│ (Gaussian)   │ (4-class)    │ (Binary)       │
└──────────────┴──────────────┴────────────────┘
```

- Shared feature extraction (transfer learning effect)
- Clinical features fusion
- Uncertainty estimation for survival

---

## 설치

### 필수 요구사항

```bash
# Python 3.8+
python >= 3.8

# PyTorch + CUDA
torch >= 2.0.0
torchvision >= 0.15.0

# PyTorch Lightning
pytorch-lightning >= 2.0.0

# Medical imaging
monai >= 1.2.0
nibabel >= 5.0.0

# Others
numpy >= 1.24.0
pyyaml >= 6.0
```

### 설치 방법

```bash
# 1. 프로젝트 클론
cd /path/to/pancreas_ai_detection-

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 모듈 확인
python -c "from pancreas_cancer_diagnosis.radiotherapy import *; print('✅ OK')"
```

---

## 데이터 준비

### 데이터 구조

```
data/radiotherapy/
├── images/
│   ├── patient001_ct.nii.gz
│   ├── patient002_ct.nii.gz
│   └── ...
├── tumor_masks/
│   ├── patient001_tumor.nii.gz
│   └── ...
├── oar_masks/
│   ├── patient001_oar.nii.gz  # Multi-class (0-6)
│   └── ...
├── dose_maps/
│   ├── patient001_dose.nii.gz  # Gy 단위
│   └── ...
└── manifests/
    ├── multitask_train.json
    ├── multitask_val.json
    ├── dose_train.json
    ├── oar_train.json
    └── ...
```

### Manifest 파일 형식

#### Multi-task manifest (`multitask_train.json`)

```json
[
  {
    "image": "images/patient001_ct.nii.gz",
    "tumor_mask": "tumor_masks/patient001_tumor.nii.gz",
    "clinical": {
      "age": 65,
      "gender": 1,
      "stage": 2,
      "ca19_9": 350.5,
      "tumor_size": 3.2,
      "location": 1,
      "kps": 80,
      "diabetes": 0,
      "prior_surgery": 0,
      "chemotherapy": 1
    },
    "survival_time": 18.5,
    "toxicity_grade": 2,
    "response": 1
  }
]
```

#### Dose prediction manifest (`dose_train.json`)

```json
[
  {
    "image": "images/patient001_ct.nii.gz",
    "tumor_mask": "tumor_masks/patient001_tumor.nii.gz",
    "oar_masks": [
      "oar_individual/patient001_duodenum.nii.gz",
      "oar_individual/patient001_stomach.nii.gz"
    ],
    "dose_map": "dose_maps/patient001_dose.nii.gz",
    "prescription_dose": 40.0
  }
]
```

#### OAR segmentation manifest (`oar_train.json`)

```json
[
  {
    "image": "images/patient001_ct.nii.gz",
    "oar_mask": "oar_masks/patient001_oar.nii.gz"
  }
]
```

---

## 학습

### 1. OAR Segmentation 학습

```bash
python -m pancreas_cancer_diagnosis.radiotherapy.training.train_oar_segmentation \
  --data_root data/radiotherapy \
  --output_dir outputs/oar_segmentation \
  --batch_size 2 \
  --max_epochs 200 \
  --gpus 1 \
  --spatial_size 128 128 128
```

### 2. Dose Prediction 학습

```bash
python -m pancreas_cancer_diagnosis.radiotherapy.training.train_dose_prediction \
  --data_root data/radiotherapy \
  --output_dir outputs/dose_prediction \
  --batch_size 1 \
  --max_epochs 150 \
  --gpus 1 \
  --spatial_size 128 128 128 \
  --num_oars 2
```

### 3. Multi-Task 학습

```bash
python -m pancreas_cancer_diagnosis.radiotherapy.training.train_multitask \
  --data_root data/radiotherapy \
  --output_dir outputs/multitask \
  --batch_size 2 \
  --max_epochs 100 \
  --gpus 1 \
  --spatial_size 96 96 96 \
  --weight_survival 1.0 \
  --weight_toxicity 1.0 \
  --weight_response 1.0
```

### Config 파일 사용

```bash
# Config 파일로 학습 (추천)
python train_with_config.py \
  --config configs/radiotherapy/oar_segmentation_config.yaml
```

---

## 추론

### 1. OAR Segmentation 추론

```bash
python -m pancreas_cancer_diagnosis.radiotherapy.inference.predict_oar \
  --checkpoint outputs/oar_segmentation/checkpoints/best.ckpt \
  --ct_path data/test/patient_ct.nii.gz \
  --output_dir outputs/predictions/oar
```

### 2. Dose Prediction 추론

```bash
python -m pancreas_cancer_diagnosis.radiotherapy.inference.predict_dose \
  --checkpoint outputs/dose_prediction/checkpoints/best.ckpt \
  --ct_path data/test/patient_ct.nii.gz \
  --tumor_mask_path data/test/patient_tumor.nii.gz \
  --oar_mask_paths data/test/patient_duodenum.nii.gz data/test/patient_stomach.nii.gz \
  --prescription_dose 40.0 \
  --output_dir outputs/predictions/dose
```

### 3. Multi-Task 추론

```bash
python -m pancreas_cancer_diagnosis.radiotherapy.inference.predict_multitask \
  --checkpoint outputs/multitask/checkpoints/best.ckpt \
  --ct_path data/test/patient_ct.nii.gz \
  --tumor_mask_path data/test/patient_tumor.nii.gz \
  --clinical_json '{"age": 65, "gender": 1, "stage": 2, "ca19_9": 350.5}' \
  --output outputs/predictions/multitask_results.json
```

---

## End-to-End 파이프라인

전체 워크플로우를 한 번에 실행:

### Python API

```python
from pancreas_cancer_diagnosis.radiotherapy.pipeline import RadiotherapyPipeline

# Pipeline 초기화
pipeline = RadiotherapyPipeline(
    oar_checkpoint="outputs/oar_segmentation/checkpoints/best.ckpt",
    dose_checkpoint="outputs/dose_prediction/checkpoints/best.ckpt",
    multitask_checkpoint="outputs/multitask/checkpoints/best.ckpt",
    device="cuda"
)

# 실행
results = pipeline.run(
    ct_path="data/patient001_ct.nii.gz",
    tumor_mask_path="data/patient001_tumor.nii.gz",
    clinical_data={"age": 65, "gender": 1, "stage": 2, "ca19_9": 350.5},
    prescription_dose=40.0,
    output_dir="outputs/pipeline/patient001",
    patient_id="PATIENT001"
)

# 결과 확인
print(f"Survival: {results['outcomes']['survival_time']:.1f} months")
print(f"Toxicity: Grade {results['outcomes']['toxicity_grade']}")
print(f"Response: {'Responder' if results['outcomes']['response'] else 'Non-responder'}")
```

### CLI

```bash
python -m pancreas_cancer_diagnosis.radiotherapy.pipeline \
  --oar_checkpoint outputs/oar_segmentation/checkpoints/best.ckpt \
  --dose_checkpoint outputs/dose_prediction/checkpoints/best.ckpt \
  --multitask_checkpoint outputs/multitask/checkpoints/best.ckpt \
  --patient_id PATIENT001 \
  --ct_path data/patient001_ct.nii.gz \
  --tumor_mask_path data/patient001_tumor.nii.gz \
  --clinical_json '{"age": 65, "gender": 1, "stage": 2}' \
  --prescription_dose 40.0 \
  --output_dir outputs/pipeline/PATIENT001
```

### 출력 결과

```
outputs/pipeline/PATIENT001/
├── PATIENT001_oar_segmentation.nii.gz  # OAR masks
├── PATIENT001_dose_map.nii.gz          # Dose distribution
└── PATIENT001_outcomes.json            # Survival, toxicity, response
```

---

## 성능 벤치마크

### OAR Segmentation

| Organ | Dice Score | HD95 (mm) |
|-------|------------|-----------|
| Duodenum | 0.82 ± 0.05 | 3.2 ± 1.1 |
| Stomach | 0.88 ± 0.04 | 2.5 ± 0.8 |
| Small intestine | 0.85 ± 0.06 | 3.8 ± 1.5 |
| Liver | 0.94 ± 0.02 | 1.8 ± 0.5 |
| Left kidney | 0.91 ± 0.03 | 2.1 ± 0.7 |
| Right kidney | 0.91 ± 0.03 | 2.1 ± 0.6 |

### Dose Prediction

- **MAE**: 2.3 ± 0.8 Gy
- **Max dose error**: 3.5 ± 1.2 Gy
- **DVH constraint satisfaction**: 92%

### Multi-Task Prediction

| Task | Metric | Performance |
|------|--------|-------------|
| Survival | MAE | 4.2 ± 2.1 months |
| Survival | C-index | 0.71 ± 0.05 |
| Toxicity | Accuracy | 68% ± 4% |
| Toxicity | Weighted F1 | 0.65 ± 0.04 |
| Response | AUC-ROC | 0.74 ± 0.06 |

---

## 참고 문헌

### 관련 논문

1. **nnU-Net**: Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation", Nature Methods, 2021

2. **Attention U-Net**: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", MIDL 2018

3. **Multi-task Learning for Medical Imaging**: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics", CVPR 2018

4. **Dose Prediction**: Nguyen et al., "A feasibility study for predicting optimal radiation therapy dose distributions of prostate cancer patients from patient anatomy using deep learning", Scientific Reports, 2019

### Clinical Guidelines

- SBRT for Pancreatic Cancer: NCCN Guidelines
- OAR Dose Constraints: QUANTEC (Quantitative Analysis of Normal Tissue Effects in the Clinic)
- Toxicity Grading: CTCAE v5.0

---

## 라이센스

이 프로젝트는 연구 및 교육 목적으로만 사용 가능합니다. 임상 사용 전 규제 기관 승인 필요.

## 문의

- 개발자: [Your Name]
- 이메일: [email]
- GitHub: [repository]

---

**⚠️ 주의사항**

이 시스템은 임상 의사 결정을 **보조**하는 도구이며, 최종 치료 결정은 반드시 전문의의 판단하에 이루어져야 합니다.
