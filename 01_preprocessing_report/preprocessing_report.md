# 상세 데이터 전처리 보고서 (Detailed Preprocessing Report)

## 1. 프로젝트 개요 및 데이터 소스
- **데이터 소스**: `data/01_raw/train.csv` (4,250 rows)
- **타겟 변수**: `churn` (이탈 여부)
- **초기 상태**: 약 14.5%의 고객이 이탈한 불균형 데이터셋.

## 2. 데이터 통합 및 세정 (Data Cleaning)
- **병합 전략**: 초기에는 `train.csv`와 `test.csv`를 병합하여 일관된 인코딩을 적용했으나, `test.csv`에 실제 레이블(Label)이 없는 것을 확인하여 최종적인 모델 학습 및 검증은 레이블이 있는 4,250개의 데이터를 기반으로 재구성했습니다.
- **불필요 변수 제거**: 모델 성능에 영향을 주지 않는 고유 식별자(`id`) 및 관리용 변수(`dataset_source`)를 제거했습니다.

## 3. 피처 엔지니어링 및 인코딩 (Feature Engineering)
### 3.1. 이진 매핑 (Binary Mapping)
- 문자열로 구성된 이진 변수들을 수치형(1/0)으로 변환하여 모델의 수렴 속도와 연산 효율을 높였습니다.
  - 대상: `international_plan`, `voice_mail_plan`, `churn`
  - 변환: `{'yes': 1, 'no': 0}`

### 3.2. 범주형 변수 처리 (Encoding)
- 다중 값을 가진 범주형 특성에 대해 **Label Encoding**을 적용했습니다.
  - 대상: `state` (주 단위 정보), `area_code` (지역 번호)
  - 특징: CatBoost 모델의 경우 자체적으로 범주형 변수를 처리할 수 있으나, 전처리 파이프라인의 일반성을 위해 수치형 변환을 우선적으로 수행했습니다.

## 4. 이상치 처리 (Outlier Handling)
- **탐지 기법**: IQR (Interquartile Range) 방식 적용.
- **결정**: 특정 지표(사용 시간, 요금 등)에서 발생하는 극단적인 값들이 실제 고객의 사용 패턴(Heavy User)을 반영하고 있다고 판단하여, **최종 파이프라인에서는 이상치를 제거하지 않고 보존**했습니다. 이는 이탈 징후를 보이는 고유 패턴을 유지하기 위함입니다.

## 5. 데이터 분할 전략 (Data Splitting Strategy)
- **방식**: **층화 분할 (Stratified Split)**
- **분할 비율**: Train 85% / Test 15%
- **표본 수**:
  - **학습용 (Train Set)**: 3,612개
  - **검증용 (Test Set)**: **638개**
- **정당성**: 클래스 불균형이 있는 데이터셋에서 학습과 테스트의 타겟 분포를 동일하게 유지함으로써, 평가 결과가 특정 클래스에 편향되지 않도록 보장했습니다. **Support 638개**는 모든 벤치마킹 실험의 공통 기준으로 사용되었습니다.

## 6. 클래스 불균형 해결 (Imbalance Mitigation)
### 6.1. 샘플링 파이프라인 (Sampling Pipeline)
- 불균형 문제를 해결하기 위해 다양한 샘플링 기법을 적용 및 실험했습니다.
  - **SMOTE**: 소수 클래스 데이터를 합성하여 증폭.
  - **SMOTE-Tomek**: SMOTE 후 Tomek-Links를 제거하여 데이터 경계를 명확화.
  - **SMOTE-ENN**: SMOTE 후 ENN(Edited Nearest Neighbors)을 사용하여 데이터 노이즈 제거.

### 6.2. 최종 선택: Class Weighting
- 샘플링 데이터가 모델의 과적합(Overfitting)을 유발할 수 있음을 확인하여, 최종적으로는 **원본 데이터를 유지하되 CatBoost의 `scale_pos_weight` (또는 `balanced` 가중치)를 사용**하는 방식을 선택했습니다. 이는 모델이 소수 클래스(이탈 고객)에 대해 더 큰 가중치를 두어 학습하게 함으로써 재현율(Recall)을 크게 향상시켰습니다.