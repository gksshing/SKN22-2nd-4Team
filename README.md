
# SKN22-2nd-4Team Project

# 📡 Telecom Customer Churn Prediction Project

**통신사 고객 이탈 예측 및 데이터 기반 의사결정 분석**

---

## 1. 프로젝트 개요

본 프로젝트는 통신사 고객 데이터를 기반으로 **고객 이탈(Churn)을 예측**하고,

단순 예측 정확도 향상을 넘어 **실제 비즈니스 의사결정에 활용 가능한 인사이트 도출**을 목표로 합니다.

특히 다음 질문에 답하는 것을 핵심 목표로 설정했습니다.

- 고객 이탈은 어떠한 원인으로 발생하는가?
- 이탈 확률이 높은 고객 그룹은 무엇인가?
- 이탈을 사전에 관리하면 재무적으로 어떤 효과가 발생하는가?

---

## 2. 데이터 개요

-**데이터 소스**: 통신사 고객 사용 이력 데이터

-**관측 단위**: 고객(Customer) 단위

-**총 데이터 수**: 4,250 rows

-**Target 변수**:

  -`churn` (1 = 이탈, 0 = 유지)

-**데이터 특성**:

- 이탈 고객 비율 약 **14~15%**
- 클래스 불균형 존재

---

## 3. 프로젝트 구조

```bash

├──01_preprocessing_report

│   └──preprocessing_report.md      # 데이터 전처리 상세 보고서

│

├──02_training_report

│   ├──benchmark_original.csv

│   ├──benchmark_smote.csv

│   ├──benchmark_smote_tomek.csv

│   ├──benchmark_smote_enn.csv

│   ├──roc_curve_original.png

│   ├──roc_curve_smote.png

│   ├──roc_curve_smote_tomek.png

│   ├──roc_curve_smote_enn.png

│   └──training_report.md            # 모델 학습 및 비교 보고서

│

├──03_trained_model

│   ├──churn_model.cbm               # 최종 학습 CatBoost 모델

│   ├──features.pkl                  # 사용 feature 목록

│   └──mean_values.pkl               # 결측 보정 기준값

│

├──data

│   ├──01_raw                         # 원본 데이터

│   ├──03_resampled                  # 샘플링 실험 데이터

│   ├──04_results                    # 실험 결과 저장

│   └──05_optimized                  # 최종 최적화 데이터

│

├──notebooks

│   ├──EDA.ipynb                     # 탐색적 데이터 분석

│   ├──EDA.md

│   └──lee_modeling.ipynb             # 모델링 실험 노트북

│

├──presentation_assets               # 발표 자료

│

└──src

    ├──data

    ├──models

    └──visualization

```

---

## 4. 데이터 전처리 요약

### 🚿데이터 세정

- 고유 식별자(`id`) 및 관리용 변수 제거
- 모델 학습에 필요한 레이블이 있는 데이터(4,250 rows)만 사용

### 💁‍♂️ 인코딩 전략

-**이진 변수(Binary Features)**

- 대상: `international_plan`, `voice_mail_plan`, `churn`
- 변환 방식:

  `yes / no` → `1 / 0`
- 목적: 모델 수렴 속도 개선 및 연산 효율 향상

-**범주형 변수(Categorical Features)**

- 대상: `state`, `area_code`
- 처리 방식: Label Encoding 적용
- 참고: CatBoost의 자체 범주형 처리 기능을 고려하였으나,

  전처리 파이프라인의 일관성을 위해 수치형 변환을 우선 적용

---

## 5. 이상치 및 클래스 불균형 처리

### 🚮 이상치 처리 (Outlier Handling)

-**탐지 기법**: IQR (Interquartile Range)

- 사용량 및 요금 관련 극단값은 실제 고객의 **Heavy User 패턴**을 반영한다고 판단
- 이탈 신호가 포함된 중요 패턴 보존을 위해

  **최종 모델에서는 이상치를 제거하지 않고 유지**

### 🚩 클래스 불균형 대응 (Imbalance Mitigation)

-**검토한 방법**

- SMOTE
- SMOTE-Tomek
- SMOTE-ENN

-**문제점**

- 샘플링 기반 접근은 일부 모델에서 과적합(Overfitting) 유발 가능성 확인

-**최종 선택**

- 데이터 증강 없이 **Class Weighting 방식 채택**
- CatBoost의 `scale_pos_weight` 또는 `balanced` 옵션 활용

-**효과**

- 소수 클래스(이탈 고객)에 대한 재현율(Recall) 유의미한 개선
- 과적합 최소화

---

## 6. 모델 학습 및 성능 요약 (Training Summary)

### 모델링 접근

- 다양한 머신러닝 알고리즘을 대상으로 **벤치마킹 실험**을 수행하여 성능을 비교
- 실험 대상:

  - Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost
  - ANN, SVM, Logistic Regression
- 모든 모델은 **동일한 Test Set (Support 638)** 기준으로 평가

---

### 최종 모델: CatBoost

**CatBoost + One-Hot Encoding(OHE)** 조합이 가장 안정적이고 실전적인 성능을 기록하여 최종 모델로 선정

#### 선정 이유

- 범주형 변수의 카디널리티가 낮아 OHE 적용 시 이탈 패턴 분리 효과가 큼
- 샘플링(SMOTE 계열) 및 클래스 가중치 적용 없이도 높은 성능 확보
- 원본 데이터 분포를 유지하여 오탐(False Positive) 최소화

---

### 하이퍼파라미터 최적화

-**최적화 도구**: Optuna

-**전처리 방식**: One-Hot Encoding / Class Weight 미적용

-**주요 최적 파라미터**

  -`depth`: 8

  -`bagging_temperature`: 0.51

  -`colsample_bylevel`: 0.52

---

### 최종 성능 지표 (Test Set 기준)

| Metric        | Value  | 설명 |

|--------------|--------|------|

| **F1-Score** | **0.88** | 정밀도·재현율 균형 |

| **Recall**   | **0.80** | 실제 이탈자 90명 중 72명 검출 |

| **Precision**| **0.97** | 오탐 2건으로 매우 높은 신뢰도 |

| **ROC AUC**  | **0.91** | 전반적 분류 성능 |

| **Threshold**| **0.36** | 비즈니스 효율 최적 임계값 |

---

### 핵심 결론

- 불균형 데이터 환경에서도 **인위적인 데이터 보정 없이** 높은 성능 달성
- 원-핫 인코딩과 CatBoost 조합이 이탈 고객의 패턴을 가장 명확히 포착
- 높은 정밀도를 기반으로 **실제 이탈 관리 시 오탐 비용 최소화 가능**
- 향후 신규 데이터 유입 시 피처 영향도 모니터링을 통해 지속적 고도화 예정

---

## 7. 프로젝트 특징

- 단순 예측 정확도 중심의 모델링이 아닌,

  **데이터 전처리 및 클래스 불균형 대응 전략의 효과를 비교·검증** 하는 데 중점을 둔 프로젝트
- 모든 모델 실험은 **동일한 테스트 셋(Test Set)** 을 기준으로 평가하여

  샘플링 기법 및 가중치 전략 간의 성능 차이를 공정하게 비교
- 샘플링 기반 접근(SMOTE 계열)과

  **Class Weighting 기반 접근을 병렬적으로 실험** 하여

  재현율(Recall)과 일반화 성능 간의 균형을 검증
- 모델 결과를 단순 예측 성능으로 끝내지 않고,

  이후 **전략 시뮬레이션 및 대시보드 형태로 확장 가능한 구조** 로 설계

-**실제 비즈니스 의사결정(이탈 관리 전략 수립)** 에 활용될 수 있도록

  전체 파이프라인을 구성
