# 🏥 췌장암 방사선 치료 AI 시스템

**췌장암 SBRT를 위한 AI 기반 방사선 치료 계획 및 결과 예측 시스템**

방사선 치료 계획을 자동화하고, 치료 결과를 예측하며, 췌장암 정위체부 방사선치료(SBRT)의 선량 분포를 최적화하는 종합 딥러닝 시스템입니다.

---

## 📋 개요

이 시스템은 췌장암 방사선 치료를 위한 3가지 핵심 AI 모델을 제공합니다:

1. **OAR Segmentation** - 주요 장기 자동 세그멘테이션 (7개 장기)
2. **Dose Prediction** - DVH 제약 조건을 고려한 최적 3D 선량 분포 예측
3. **Multi-task Learning** - 생존 시간, 독성, 치료 반응 동시 예측

### 임상적 목표

- ✅ **치료 계획 최적화**: 자동화된 치료 계획으로 시간 단축
- ✅ **독성 예측**: GI 독성(십이지장) 위험 조기 평가
- ✅ **예후 예측**: 환자별 맞춤형 생존 및 반응 예측
- ✅ **Multi-Center 강건성**: 다양한 병원 및 스캐너에서 작동

---

## 🚀 주요 기능

### 고급 AI 모델

- **OAR Segmentation**: nnU-Net 기반 아키텍처 + Deep supervision
- **Dose Prediction**: Attention gate를 사용한 3D U-Net + DVH loss
- **Multi-task Learning**: Shared encoder + Uncertainty quantification

### Multi-Site 지원

- ✅ Site-specific normalization (병원별 통계)
- ✅ Histogram matching (도메인 적응)
- ✅ Weighted sampling (데이터 불균형 처리)
- ✅ Adaptive preprocessing (스캐너 변동성 대응)

### End-to-End 파이프라인

- 전체 워크플로우 원클릭 실행
- Python API + CLI 지원
- 자동 품질 관리
- 상세한 로깅

---

## 🏗️ 프로젝트 구조

```
pancreas_cancer_diagnosis/
└── radiotherapy/                    # 방사선 치료 AI 모듈
    ├── models/                      # 신경망 아키텍처
    │   ├── base.py                  # 베이스 모델 클래스
    │   ├── multi_task_model.py     # Multi-task (생존 + 독성 + 반응)
    │   ├── dose_prediction.py      # 선량 분포 예측
    │   └── oar_segmentation.py     # OAR 세그멘테이션
    │
    ├── data/                        # 데이터 로딩 및 전처리
    │   ├── datasets.py              # PyTorch 데이터셋 (3종류)
    │   ├── datamodule.py            # Lightning DataModule
    │   ├── transforms.py            # MONAI 전처리 (6종류)
    │   ├── preprocessing.py         # 고급 전처리 (5개 Transform)
    │   └── multisite_datamodule.py  # Multi-site 데이터 처리
    │
    ├── training/                    # 학습 스크립트
    │   ├── train_multitask.py
    │   ├── train_dose_prediction.py
    │   └── train_oar_segmentation.py
    │
    ├── inference/                   # 추론 스크립트
    │   ├── predict_multitask.py
    │   ├── predict_dose.py
    │   └── predict_oar.py
    │
    ├── pipeline.py                  # End-to-end 파이프라인
    └── README.md                    # 모듈 상세 문서

configs/radiotherapy/                # 설정 파일
├── multitask_config.yaml
├── dose_prediction_config.yaml
├── oar_segmentation_config.yaml
└── pipeline_config.yaml

docs/
└── MULTI_SITE_GUIDE.md             # Multi-site 학습 가이드
```

---

## 📦 설치

### 필수 요구사항

```bash
python >= 3.8
torch >= 2.0.0
pytorch-lightning >= 2.0.0
monai >= 1.2.0
nibabel >= 5.0.0
SimpleITK >= 2.2.0
```

### 설치 방법

```bash
# 저장소 클론
git clone <repository-url>
cd pancreas_ai_detection-

# 의존성 설치
pip install -r requirements.txt

# 설치 확인
python -c "from pancreas_cancer_diagnosis.radiotherapy import *; print('✅ 설치 완료')"
```

---

## 🎯 빠른 시작

### 1. 학습

#### OAR Segmentation 학습

```bash
python -m pancreas_cancer_diagnosis.radiotherapy.training.train_oar_segmentation \
  --data_root data/radiotherapy \
  --batch_size 2 \
  --max_epochs 200
```

#### Dose Prediction 학습

```bash
python -m pancreas_cancer_diagnosis.radiotherapy.training.train_dose_prediction \
  --data_root data/radiotherapy \
  --batch_size 1 \
  --max_epochs 150
```

#### Multi-task Learning 학습

```bash
python -m pancreas_cancer_diagnosis.radiotherapy.training.train_multitask \
  --data_root data/radiotherapy \
  --batch_size 2 \
  --max_epochs 100
```

### 2. 추론 (End-to-End)

```bash
python -m pancreas_cancer_diagnosis.radiotherapy.pipeline \
  --oar_checkpoint outputs/oar/best.ckpt \
  --dose_checkpoint outputs/dose/best.ckpt \
  --multitask_checkpoint outputs/multitask/best.ckpt \
  --ct_path patient_ct.nii.gz \
  --tumor_mask_path tumor.nii.gz \
  --clinical_json '{"age": 65, "stage": 2}' \
  --prescription_dose 40.0 \
  --output_dir outputs/results
```

### 3. Python API

```python
from pancreas_cancer_diagnosis.radiotherapy.pipeline import RadiotherapyPipeline

# 파이프라인 초기화
pipeline = RadiotherapyPipeline(
    oar_checkpoint="outputs/oar/best.ckpt",
    dose_checkpoint="outputs/dose/best.ckpt",
    multitask_checkpoint="outputs/multitask/best.ckpt"
)

# 예측 실행
results = pipeline.run(
    ct_path="patient_ct.nii.gz",
    tumor_mask_path="tumor.nii.gz",
    clinical_data={"age": 65, "stage": 2},
    prescription_dose=40.0,
    output_dir="outputs/patient001"
)

# 결과 확인
print(f"생존 시간: {results['outcomes']['survival_time']:.1f}개월")
print(f"독성 등급: Grade {results['outcomes']['toxicity_grade']}")
print(f"치료 반응: {'반응자' if results['outcomes']['response'] else '비반응자'}")
```

---

## 🌐 Multi-Site 학습

여러 병원/센터의 데이터로 학습하는 경우:

```python
from pancreas_cancer_diagnosis.radiotherapy.data.multisite_datamodule import (
    MultiSiteMultiTaskDataModule
)

# Site 설정
site_configs = [
    {"site_name": "병원_A", "train_manifest": "...", "weight": 0.4},
    {"site_name": "병원_B", "train_manifest": "...", "weight": 0.3},
    {"site_name": "병원_C", "train_manifest": "...", "weight": 0.3}
]

# DataModule 생성
datamodule = MultiSiteMultiTaskDataModule(
    data_root="data/radiotherapy",
    site_configs=site_configs,
    use_site_normalization=True,
    use_weighted_sampling=True
)

# 학습
trainer.fit(model, datamodule)
```

**자세한 내용은 [Multi-Site 가이드](docs/MULTI_SITE_GUIDE.md)를 참고하세요.**

---

## 📖 문서

- [Radiotherapy 모듈 README](pancreas_cancer_diagnosis/radiotherapy/README.md) - 모듈 상세 문서
- [Multi-Site 학습 가이드](docs/MULTI_SITE_GUIDE.md) - Multi-center 데이터 처리
- [설정 파일](configs/radiotherapy/) - YAML config 예제

---

## 🔬 기대 효과

1. **시간 효율성**: 수동 계획 대비 치료 계획 시간 단축
2. **안전성**: GI 독성 위험 조기 예측
3. **개인화**: 환자별 맞춤형 결과 예측
4. **일관성**: 계획자 간 변동성 감소
5. **범용성**: 다양한 병원 환경에서 활용 가능

---

## ⚠️ 주의사항

이 시스템은 임상 의사 결정을 **보조**하는 도구입니다. 최종 치료 결정은 반드시 전문의의 판단하에 이루어져야 합니다.

---

## 📄 라이센스

이 프로젝트는 연구 및 교육 목적으로만 사용 가능합니다. 임상 사용 시 규제 기관 승인이 필요합니다.

---

## 📧 문의

- Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 문서: `/docs` 및 모듈 README 참조

---

## 🙏 참고 자료

임상 가이드라인 기반:
- NCCN Guidelines for Pancreatic Cancer
- QUANTEC (주요 장기 선량 제약)
- CTCAE v5.0 (독성 등급 분류)

---

**최종 업데이트**: 2025-11-05
