# optimization.py 코드 상세 분석

이 문서는 `src/models/optimization.py` 파일의 구조, 기능, 및 실행 로직에 대한 상세한 설명을 담고 있습니다.

## 1. 개요 (Overview)
이 스크립트는 Hyperparameter Optimization (하이퍼파라미터 최적화) 도구인 **Optuna**를 사용하여, `XGBoost`와 `LightGBM` 모델의 성능을 극대화하는 최적의 파라미터를 탐색합니다. 'Original' 데이터셋(가중치 적용)을 사용하여 F1-Score를 최대화하는 방향으로 학습하며, 최적화된 모델에 대해 **임계값 튜닝(Threshold Tuning)**까지 수행하여 최종적인 이탈 예측 성능을 도출합니다.

## 2. 주요 설정 (Configuration)
- **DATASET_NAME**: `'original'`
  - SMOTE 등의 샘플링 데이터 대신, 원본 데이터에 Class Weight를 적용하는 방식이 선정되었음을 반영합니다.
- **N_TRIALS**: `10`
  - Optuna의 시도 횟수입니다. (시연 및 테스트 속도를 위해 낮게 잡혀 있으나, 실제 운영 시에는 더 높여야 할 수 있습니다.)
- **경로**: `data\03_resampled` (입력), `data\05_optimized` (결과 출력).

## 3. 핵심 컴포넌트

### 3.1. `find_optimal_threshold(y_true, y_prob)`
- **기능**: 예측 확률(`y_prob`)에 대해 0.01부터 0.99까지의 모든 임계값을 테스트하여, **F1-Score가 가장 높은 임계값(Threshold)**을 찾아냅니다.
- **목적**: 불균형 데이터에서는 기본 임계값(0.5)이 최적이 아닐 확률이 높으므로, 이를 조정하여 Recall과 Precision의 균형을 맞춥니다.

### 3.2. `ModelOptimizer` 클래스
Optuna의 `study.optimize`에 전달될 목적 함수(`objective`)를 캡슐화한 클래스입니다.

#### `__init__`
- 학습 데이터와 모델 이름을 입력받습니다.
- **scale_pos_weight 계산**: 데이터의 불균형 비율(Negative/Positive)을 계산하여 XGBoost의 가중치 파라미터로 사용할 준비를 합니다.

#### `objective(trial)`
- **탐색 공간(Search Space)**:
  - **XGBoost**: `lambda`, `alpha`, `subsample`, `colsample_bytree`, `max_depth`, `eta`, `gamma` 등.
  - **LightGBM**: `num_leaves`, `max_depth`, `lambda_l1`, `lambda_l2`, `feature_fraction` 등.
- **고정 파라미터**:
  - `booster='dart'` (Dropouts meet Multiple Additive Regression Trees): 과적합 방지에 강한 부스팅 방식 사용.
  - `class_weight='balanced'` (LGBM) 또는 `scale_pos_weight` (XGB): 불균형 처리를 위한 필수 설정.
- **검증(Validation)**:
  - `StratifiedKFold(n_splits=3)`를 사용하여 교차 검증을 수행합니다.
  - 평가 지표로 `f1` 점수의 평균값을 반환하여 Optuna가 이를 최대화하도록 유도합니다.

## 4. 메인 실행 흐름 (`run_optimization`)

1.  **데이터 준비**:
    - `load_data`를 통해 Train/Test 데이터를 로드합니다.
    - 전체 데이터셋에 대한 `scale_pos_weight`를 다시 한 번 계산합니다.

2.  **최적화 루프 (For Loop)**:
    - [`XGBoost`, `LightGBM`] 두 모델에 대해 순차적으로 최적화를 진행합니다.
    - `optuna.create_study` -> `study.optimize` 흐름으로 하이퍼파라미터 탐색을 수행합니다.
    - 가장 높은 검증 F1 점수를 기록한 파라미터 조합(`best_params`)을 찾습니다.

3.  **최종 모델 학습 (Final Model Training)**:
    - 탐색된 `best_params`에 `n_estimators=1000` (충분한 학습량) 등을 추가하여 최종 모델을 정의합니다.
    - 전체 **Train Set**으로 모델을 재학습시킵니다.

4.  **임계값 최적화 및 평가**:
    - **Test Set**에 대해 확률(`predict_proba`)을 예측합니다.
    - `find_optimal_threshold`를 호출하여 Test Set 기준 최적 임계값을 찾습니다. ( *참고: 엄밀하게는 Valid Set으로 찾아야 하나, 여기서는 최종 성능 확인용으로 Test Set을 활용하고 있습니다.* )
    - 최적화된 임계값을 적용하여 최종 `0/1` 예측 결과를 생성합니다.

5.  **결과 저장**:
    - 모델별 최적 파라미터, ROC AUC, 최적 임계값, 분류 리포트(Classification Report)를 텍스트 파일로 저장합니다.
    - 저장 경로: `data\05_optimized\{model_name}_optimization_report.txt`

## 요약
이 파일은 단순한 Grid Search보다 효율적인 **Bayesian Optimization (Optuna)** 기법을 도입하여 모델 성능의 한계치를 끌어올리는 심화 학습 모듈입니다. 특히 **Class Weight + DART Boosting + Threshold Tuning**의 3단계 전략을 통해 불균형 데이터에서의 성능 극대화를 목표로 하고 있다는 점이 특징입니다.
