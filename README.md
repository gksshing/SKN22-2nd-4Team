# 📡 Telecom Customer Churn Diagnosis Project

**"단순 예측을 넘어, 비즈니스 가치를 창출하는 이탈 방지 솔루션"**

---

## 1. 프로젝트 개요 (Executive Summary)

### **🎯 목표**
통신사 고객 데이터를 분석하여 **이탈(Churn)을 사전에 예측**하고, 이를 방지하기 위한 **실질적인 비즈니스 전략**을 수립합니다. 단순한 정확도 경쟁이 아닌, **"왜 이탈하는가?"**에 대한 해석과 **"어떻게 막을 것인가?"**에 대한 액션 플랜을 제시하는 데 초점을 맞췄습니다.

### **🔍 핵심 질문**
1. 이탈을 유발하는 **핵심 원인(Risk Factor)**은 무엇인가?
2. 이탈 확률이 높은 **고객군(Segment)**은 누구인가?
3. 이탈을 방어했을 때 얻을 수 있는 **재무적 가치(ROI)**는 얼마인가?

---

## 2. 데이터 개요 (Data Description)

- **데이터 소스**: 통신사 고객 사용 이력 데이터 (4,250 rows)
- **Target 변수**: `churn` (1 = 이탈, 0 = 유지)
- **데이터 특성**:
  - **불균형 데이터**: 이탈률 약 **14%**
  - **주요 피처**: 주간/야간 통화량, 국제전화 요금, 고객센터 통화 횟수, 요금제 가입 여부 등

---

## 3. 핵심 방법론 (Methodology)

본 프로젝트는 데이터 전처리와 모델링 기법의 정교한 조합을 통해 최적의 성능을 도출했습니다 상세 내용은 `01_preprocessing_report` 및 `02_training_report`를 참조하십시오.

### **🛠 1. 데이터 전처리 (Preprocessing)**
- **이상치(Outlier) 보존**: 'Heavy User'의 패턴(높은 통화량 등)이 실제 이탈과 밀접한 연관이 있음을 확인하여, 이상치를 인위적으로 제거하지 않고 보존했습니다.
- **인코딩(Encoding)**: CatBoost의 자체 처리 방식 대신 **원-핫 인코딩(One-Hot Encoding)**을 적용하여, 범주형 변수의 영향력을 독립적인 피처로 분리·강화했습니다.

### **⚖️ 2. 불균형 데이터 처리 (Handling Imbalance)**
- **실험**: SMOTE, SMOTE-Tomek, SMOTE-ENN 등 다양한 오버샘플링 기법과 Class Weight 적용을 비교 실험했습니다.
- **최종 선택**: **"No Class Weights + Original Data"**
  - **이유**: 인위적인 데이터 증강이나 가중치 부여가 오히려 정밀도(Precision)를 떨어뜨리고 노이즈를 유발함을 확인했습니다. 
  - **결과**: 원본 데이터의 패턴을 그대로 학습시켰을 때 **F1-Score 0.88**이라는 가장 우수한 성과를 달성했습니다.

### **🤖 3. 모델링 (Modeling)**
- **최종 모델**: **CatBoost Classifier**
- **최적화**: Optuna를 활용해 `depth`, `bagging_temperature` 등 핵심 파라미터를 최적화했습니다.

---

## 4. 최종 성과 (Performance)

테스트 데이터(Support 638명) 기준, 실전 비즈니스에 즉시 투입 가능한 수준의 **높은 신뢰도**를 확보했습니다.

| 지표 (Metric) | 결과 (Result) | 비즈니스 의미 |
| :--- | :--- | :--- |
| **ROC AUC** | **0.91** | 우수한 판별력 (이탈자와 비이탈자를 정확히 구분) |
| **Precision** | **0.97** | **오탐(False Positive) 최소화**. 잘못된 타겟팅으로 인한 마케팅 비용 낭비 방지 |
| **Recall** | **0.80** | **실제 이탈자의 80%를 사전에 탐지**하여 방어 기회 확보 |
| **F1-Score** | **0.88** | 정밀도와 재현율의 이상적인 균형 달성 |

---

## 5. 프로젝트 구조 (Directory Structure)

```bash
📦 SKN22-2nd-4Team
 ┣ 📂 01_preprocessing_report   # 데이터 전처리 상세 보고서
 ┣ 📂 02_training_report        # 모델 실험 및 성능 평가 보고서
 ┣ 📂 03_trained_model          # 최종 학습된 모델 (churn_model.cbm)
 ┣ 📂 data                      # 데이터 저장소 (Raw, Resampled, Optimized)
 ┣ 📂 notebooks                 # EDA 및 실험용 Jupyter Notebook
 ┣ 📂 presentation_assets       # 발표 및 리포팅용 시각화 자료
 ┣ 📂 src                       # 소스 코드
 ┃ ┣ 📂 data                    # 전처리 파이프라인
 ┃ ┣ 📂 experiments             # 실험 파이프라인      
 ┃ ┣ 📂 models                  # 모델 학습, 최적화, 벤치마크 스크립트
 ┃ ┗ 📂 visualization           # 시각화 스크립트
 ┣ 📜 app.py                    # Streamlit 기반 대시보드 애플리케이션
 ┣ 📜 requirements.txt          # 의존성 패키지 목록
 ┗ 📜 README.md                 # 프로젝트 문서 (Current)
```

---

## 6. 사용 가이드 (How to Use)

### **1. 🚀 대시보드 실행 (Streamlit App)**
본 프로젝트는 이해관계자가 결과를 직관적으로 확인하고 시뮬레이션할 수 있는 **웹 대시보드**를 제공합니다.

```bash
# 필수 패키지 설치
pip install -r requirements.txt

# 대시보드 실행
streamlit run app.py
```

### **2. 🧪 모델 최적화 및 실험 (Optimization)**
새로운 데이터나 파라미터로 모델을 다시 최적화하려면 아래 스크립트를 실행합니다.

```bash
# CatBoost 최적화 (Raw Data + No Class Weights)
python src/models/optimization.py
```

---

## 7. 기대 효과 및 결론

1. **마케팅 효율 극대화**: **97%의 높은 정밀도**로, 정말로 떠날 것 같은 고객에게만 혜택을 집중하여 예산을 절감합니다.
2. **맞춤형 방어 전략**: VIP, 국제전화 사용자, 불만 고객 등 **세그먼트별 맞춤 전략**을 통해 이탈을 효과적으로 차단합니다.
3. **지속 가능한 시스템**: 단순 일회성 분석이 아닌, **데이터 적재 → 학습 → 시각화 → 액션**으로 이어지는 선순환 구조를 구축했습니다.
