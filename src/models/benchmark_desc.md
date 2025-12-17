# benchmark.py 코드 상세 분석

이 문서는 `src/models/benchmark.py` 파일의 구조, 기능, 및 실행 로직에 대한 상세한 설명을 담고 있습니다.

## 1. 개요 (Overview)
이 스크립트는 전처리 및 샘플링 과정을 거쳐 생성된 **4가지 데이터셋(Original, SMOTE, SMOTE-Tomek, SMOTE-ENN)**에 대해 **8가지 머신러닝 모델**을 학습시키고 성능을 평가하는 벤치마킹 모듈입니다. 평가 결과(Precision, Recall, F1-score, ROC-AUC)를 CSV로 저장하고, ROC Curve를 시각화하여 저장합니다.

## 2. 주요 설정 (Configuration)
- **데이터 경로**: `data\03_resampled`에서 학습/테스트 데이터를 불러옵니다.
- **결과 저장 경로**: `data\04_results`에 결과 CSV와 그래프 이미지를 저장합니다.
- **RANDOM_STATE**: 모든 모델에 동일한 시드(42)를 적용하여 공정한 비교를 보장합니다.

## 3. 함수 별 기능 설명

### 3.1. `load_dataset(dataset_name)`
- **기능**: 지정된 샘플링 전략(`dataset_name`)에 해당하는 학습 데이터와 공통 테스트 데이터를 로드합니다.
- **특징**:
  - `y` 값(Target)은 모델 입력 형식을 맞추기 위해 `ravel()`을 사용하여 1차원 배열로 변환합니다.

### 3.2. `get_models(use_class_weights=False)`
- **기능**: 실험에 사용할 모델들의 인스턴스를 생성하여 딕셔너리로 반환합니다.
- **포함된 모델**:
  1. **Tree-based**: DecisionTree, RandomForest, XGBoost, LightGBM, CatBoost
  2. **Others**: ANN(MLP), SVM, LogisticRegression (Pipeline으로 Scaling 적용)
- **Class Weights (클래스 가중치)**:
  - `use_class_weights=True`인 경우, 불균형 해결을 위해 모델별로 가중치 파라미터(`balanced` 등)를 설정합니다.
  - **XGBoost**: `scale_pos_weight`를 약 5.9로 설정 (Negative/Positive 비율 기반).

### 3.3. `load_raw_data_for_catboost()`
- **기능**: **CatBoost 모델의 성능 극대화**를 위해 별도로 원본 데이터를 로드하는 함수입니다.
- **목적**: CatBoost는 이미 전처리(Label Encoding 등)된 데이터보다, **Raw 데이터(문자열 범주형 변수)**를 직접 처리할 때 성능이 더 뛰어납니다. 따라서 'Original' 데이터셋 실험 시에는 전처리된 CSV 대신 이 함수를 통해 Raw 데이터를 다시 불러와서 학습에 사용합니다.

### 3.4. `evaluate_models_on_dataset(dataset_name)`
- **기능**: 특정 데이터셋에 대해 모든 모델을 학습 및 평가합니다.
- **로직**:
  1. `load_dataset`으로 데이터를 불러옵니다.
  2. 'original' 데이터셋인 경우에만 `Class Weights` 기능을 활성화합니다. (이미 오버샘플링된 데이터셋에는 가중치 적용 불필요).
  3. **CatBoost 특화 로직**:
     - `dataset_name == 'original'`일 때, `load_raw_data_for_catboost`를 호출하여 Raw 데이터를 사용하고, `cat_features` 파라미터를 설정하여 학습합니다.
  4. **평가 지표 계산**:
     - `Precision`, `Recall`, `F1-score`, `ROC-AUC`를 계산합니다.
     - ROC Curve 생성을 위한 `fpr`, `tpr` 값을 저장합니다.

### 3.5. `plot_roc_curves(roc_data, dataset_name)`
- **기능**: 모든 모델의 ROC Curve를 하나의 그래프에 그려 비교합니다.
- **출력**: `roc_curve_{dataset_name}.png` 파일로 저장됩니다.

## 4. 메인 실행 흐름 (`main` 함수)

1.  **반복 실험**:
    - `datasets = ['original', 'smote', 'smote_tomek', 'smote_enn']` 리스트를 순회하며 4번의 대규모 실험을 수행합니다.
    
2.  **결과 저장**:
    - 각 데이터셋마다 `evaluate_models_on_dataset`을 호출하여 결과를 얻습니다.
    - 결과 테이블을 `benchmark_{dataset_name}.csv`로 저장합니다.
    - 결과 그래프를 `roc_curve_{dataset_name}.png`로 저장합니다.

3.  **종료**:
    - 모든 벤치마크가 완료되면 성공 메시지를 출력합니다.

## 5. 생성되는 결과물 (data/04_results 폴더)
이 스크립트 실행 시 각 데이터셋(4개) 별로 CSV 1개, PNG 1개씩 총 8개의 파일이 생성됩니다.
- 예: `benchmark_original.csv`, `roc_curve_original.png`

## 요약
이 파일은 다양한 불균형 처리 전략(샘플링 vs 가중치)과 다양한 알고리즘 간의 성능을 객관적으로 비교하기 위한 자동화된 실험 도구입니다. 특히 CatBoost의 경우 고유 기능을 활용하기 위한 별도 경로를 마련해두는 등 정교하게 구현되어 있습니다.
