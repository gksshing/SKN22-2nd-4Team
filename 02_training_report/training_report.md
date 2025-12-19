# 상세 인공지능 학습 결과 보고서 (Detailed Training Report)

## 1. 모델링 프로세스 개요
본 프로젝트에서는 고객 이탈을 사전에 탐지하기 위해 다양한 머신러닝 알고리즘을 벤치마킹하고, 최적의 모델을 선정하여 파라미터 튜닝을 진행했습니다.

## 2. 모델 알고리즘 비교 (Benchmark)
- **실험 대상**: Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, ANN, SVM, Logistic Regression
- **평가 기준**: 동일한 Test Set (Support 638)에서 F1-Score 및 Recall 지표 비교
- **주요 결과**:
  - 대부분의 트리 기반 앙상블 모델(XGBoost, CatBoost, LightGBM)이 약 0.88의 높은 F1-Score를 기록했습니다.
  - 특히 **CatBoost**는 불균형 데이터에서도 안정적인 재현율(Recall)을 보여 최종 모델 후보로 선정되었습니다.

## 3. 최종 모델 선정: CatBoost
### 3.1. 선정 사유 (Rationale)
1.  **범주형 변수의 강력한 처리**: `state`, `area_code`와 같은 범주형 변수를 원-핫 인코딩 없이 내부적으로 최적화하여 정보 손실을 최소화합니다.
2.  **클래스 불균형 해결 능력**: `auto_class_weights='Balanced'` 옵션을 통해 소수 클래스(이탈 고객)에 대한 가중치를 자동으로 조절, 높은 재현율을 확보할 수 있습니다.
3.  **성능의 견고함**: 오버샘플링(SMOTE 등) 데이터보다 원본 데이터의 가중치를 조절했을 때 더 정교한 예측 성능(Precision 유지)을 보였습니다.

## 4. 하이퍼파라미터 최적화 (Optuna Tuning)
- **최적화 도구**: Optuna 프레임워크 사용
- **시도 횟수 (Trials)**: 10회 (Native Categorical Handling 적용)
- **최적 파라미터**:
  - `depth`: 10
  - `bagging_temperature`: 0.37
  - `colsample_bylevel`: 0.87
- **데이터 처리**: `state`, `area_code` 등을 인코딩 없이 CatBoost Native Handling으로 처리

## 5. 최종 성능 지표 (Final Evaluation)
최종 학습된 CatBoost 모델의 테스트 데이터(Support 638) 기준 성능입니다.

| 지표 (Metric) | 결과 (Result) | 비고 |
| :--- | :--- | :--- |
| **F1-Score (Optimized)** | **0.88** | 정밀도와 재현율의 조화 평균 |
| **Recall (재현율)** | **0.81** | 실제 이탈자 중 찾아낸 비율 |
| **Precision (정밀도)** | **0.96** | 이탈로 예측한 고객 중 실제 이탈 비율 |
| **ROC AUC** | **0.91** | 모델의 전반적인 분류 판별 성능 |
| **Optimal Threshold** | **0.37** | F1 최대화를 위한 확률 임계값 |

## 6. 결론 및 향후 계획
- 현재 모델은 약 91%의 판별력(AUC)과 88%의 안정적인 F1-Score를 기록하고 있습니다.
- 특히 가중치 적용을 통해 이탈 고객에 대한 민감도를 유지하면서도, 96%의 높은 정밀도를 확보하여 마케팅 비용의 효율성을 극대화했습니다.
- 범주형 데이터를 원본 그대로 처리함으로써 데이터 전처리 과정을 간소화하고 성능의 신뢰성을 높였습니다.
- 추후 실제 운영 데이터를 지속적으로 학습시켜 임계값(Threshold)을 미세 조정할 예정입니다.
