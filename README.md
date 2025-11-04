# Pancreatic Cancer Early Detection AI - Stage 1 Detection System

**AI system for early detection of pancreatic cancer in CT scans performed for other purposes**

Our goal is to detect Stage 1-2 pancreatic cancer before symptoms appear by identifying subtle changes in normal pancreatic tissue.

---

## 🏥 Clinical Background

### Why is Stage 1 Early Detection Critical?

Pancreatic cancer survival rates vary dramatically based on the stage at diagnosis:

| Stage | 5-Year Survival | Detection on CT |
|-------|----------------|-----------------|
| **Stage 1** (T1N0M0, <2cm) | **~80%** | Nearly impossible ❌ |
| **Stage 2** (T2-3, 2-4cm) | ~30% | Very difficult |
| **Stage 3-4** (locally advanced/metastatic) | **<5%** | **Most cases detected here** ✓ |

**Reality:** When pancreatic cancer is detected on CT, it's usually already Stage 3-4.

### Why is Early Detection on CT So Difficult?

1. **Contrast Issues**
   - Small tumors (<1-2cm) have minimal density difference from normal pancreatic tissue
   - Standard abdominal CT protocols are not optimized for pancreatic imaging

2. **Absence of Symptoms**
   - Stage 1-2: Nearly asymptomatic → No reason to get scanned
   - **Jaundice**: Only appears when bile duct is obstructed → Already Stage 3-4

3. **Screening Limitations**
   - Health checkups: Mainly abdominal ultrasound (pancreas visualization rate: 50-70%)
   - CT in health checkups: **<5% of cases**
   - CT is usually performed "because of symptoms" → Already too late

### When Are CTs Actually Performed?

CTs are primarily performed in these situations rather than routine health checkups:

1. **Outpatient/Emergency Room Visits for Symptoms**
   - Abdominal pain, indigestion, weight loss, jaundice
   - Evaluation of other organs (liver, gallbladder, GI tract)
   - Pancreas is only observed incidentally

2. **Regular Follow-up for Other Diseases**
   - Liver disease, kidney disease, post-colorectal cancer surgery follow-up
   - Annual CTs, but pancreas not carefully examined

3. **Trauma/Pre-surgical Evaluation**
   - Pancreas is not the primary focus

**💡 Key Insight:** Over **5 million abdominal CTs** are performed annually in South Korea alone, but the pancreas is "casually glanced at" in most cases. This is where AI creates opportunity.

---

## 🎯 Project Goals and Value

### Opportunistic Pancreatic Cancer Screening

**"Automatically analyze the pancreas in CT scans performed for other purposes"**

#### Scenario A: Incidental Finding During Symptom Evaluation
```
Patient: "I have indigestion and bloating"
Doctor: Orders abdominal CT (for stomach/gallbladder evaluation)
Radiologist: "No significant findings" (pancreas briefly reviewed)

→ AI Applied ⭐
   "Subtle texture heterogeneity detected in pancreatic body"
   "Confidence: 68%, recommend pancreas-dedicated CT or MRI"

→ Additional workup → Stage 1 (1.5cm) detected
→ Surgical resection possible, chance for cure
```

#### Scenario B: Early Detection During Regular Follow-up
```
Patient: Liver cirrhosis follow-up (CT every 6 months)

→ AI's Longitudinal Analysis ⭐
   "Analysis of 3 consecutive CTs"
   "Progressive texture change in pancreatic head region"
   "No size change, but attenuation pattern change detected"

→ Dedicated pancreas protocol CT
→ Stage 1 detected (before symptoms)
```

#### Scenario C: Large-scale Retrospective Screening
```
Past CT scans stored in hospital PACS
→ AI automatically reanalyzes (Batch Processing)
→ Identifies previously missed suspicious findings
→ Recommend follow-up testing for relevant patients
```

### Core Roles of AI

1. **Subtle Feature Detection**
   - Texture changes imperceptible to the human eye
   - Pancreatic border irregularity
   - Parenchymal attenuation heterogeneity
   - Focal pancreatic duct dilation (<3mm)

2. **Longitudinal Monitoring**
   - Automatic comparison with patient's prior CTs
   - Detection of very slow growth rates
   - Differentiation between normal variation and pathologic change

3. **Multi-hospital Domain Adaptation**
   - Different CT equipment and protocols across hospitals
   - Learning normal distributions for each institution
   - Overcoming domain shift

4. **Uncertainty Quantification**
   - Stage 1 detection inevitably has high false positives
   - Recommendations with confidence scores for surveillance
   - Differentiate "immediate additional testing" vs "recheck in 3 months"

---

## 📌 Key Features

- **Opportunistic Screening**: Automatic pancreas analysis in CTs performed for other purposes
- **Anomaly Detection**: Detection of subtle abnormal patterns after learning normal pancreas
- **Longitudinal Analysis**: Detection of progressive changes through time-series CT comparison
- **Multi-hospital Adaptation**: Domain adaptation to overcome hospital/equipment differences
- **Uncertainty Quantification**: Confidence-based recommendations for surveillance or immediate workup
- **Full Modularity**: Segmentation, classification, and anomaly detection modules run independently

---

## 🔬 CT Imaging Optimization and Technical Approach

### Pancreas-Dedicated CT Protocol (Ideal Case)

To visualize Stage 1 pancreatic cancer, more sophisticated protocols than standard abdominal CT are needed:

#### 1. Multi-phase Contrast Enhancement
```
Late Arterial Phase (Pancreatic Parenchymal Phase)
  - Timing: 40-50 seconds
  - Optimal visualization of pancreatic parenchyma
  - Maximum contrast between small tumors and normal tissue

Portal Venous Phase
  - Timing: 70-80 seconds
  - Evaluation of surrounding vessels
  - Assessment of vascular invasion

Delayed Phase (Optional)
  - Timing: 3-5 minutes
  - Detection of some hypovascular tumors
```

#### 2. Thin-slice Acquisition
```
Slice Thickness: 0.5-1mm (vs standard CT: 3-5mm)
→ Reduced partial volume effect
→ Improved small lesion detection
→ Better 3D reconstruction quality
```

#### 3. High Resolution Settings
```
Matrix: 512x512 or higher
Field of View: Focused on pancreas
Reconstruction: Multiple algorithm combinations
```

### Reality: Must Work with Standard Abdominal CT

**However, most CTs are NOT acquired this way:**
- Single phase or simple dual-phase
- 5mm slice thickness
- Pancreas is not the primary target

**Therefore, our AI must:**
- ✅ **Work with suboptimal CT protocols**
- ✅ **Be robust to various protocols**
- ✅ **Learn different equipment/settings across hospitals**

---

## 🧠 Technical Approach: Anomaly Detection

### Why Anomaly Detection?

#### Problem: Lack of Cancer Data
```
Public Datasets (NIH Pancreas-CT):
  - Normal pancreas: 82 cases ✓
  - Pancreatic cancer: 0 cases ❌

Real Clinical Data:
  - Normal/benign conditions: Hundreds of thousands
  - Stage 1-2 pancreatic cancer: Hundreds (very rare)
```

#### Solution: Perfect Learning of Normal
```
"If we learn perfectly what is normal,
 we can identify what is not normal (anomaly)"
```

### Core Ideas

#### 1. U-Net Based Autoencoder
```python
# Train only on normal pancreas
Input: Normal pancreatic CT
→ Encoder: Feature compression
→ Decoder: Reconstruct original
Output: Reconstructed CT

Loss = MSE(Input, Output)
```

**After training on normal data:**
- Normal pancreas → Perfect reconstruction (low error)
- Pancreas with cancer → Reconstruction failure (high error)
- **High error region = Anomaly = Suspicious area**

#### 2. Weighted Reconstruction Loss
```python
# Higher weight on pancreatic region
Loss = weighted_MSE(Input, Output, pancreas_mask)

Pancreas region: weight = 10.0
Background: weight = 1.0
```

**Rationale:**
- To detect even small tumors (<1cm)
- Focus on subtle changes within pancreas
- Ignore background noise

#### 3. Multi-scale Feature Analysis
```
Simultaneous analysis at multiple resolutions:
- High resolution: Small tumors (<1cm)
- Medium resolution: Texture patterns
- Low resolution: Overall morphological changes
```

#### 4. Temporal Consistency (Future Plan)
```
Compare with patient's prior CTs:
- t0: Normal (baseline)
- t1: Subtle change (AI detects)
- t2: Clear change (confirmed)

→ Learn progressive change patterns
→ Reduce false positives
```

### Expected Output Example

```json
{
  "patient_id": "P001234",
  "scan_date": "2025-10-15",
  "anomaly_detected": true,
  "anomaly_score": 0.73,
  "recommendation": "Recommend pancreas-dedicated CT or MRI",
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

## 🏗️ Project Structure

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
