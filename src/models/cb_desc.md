# cb.py 코드 상세 분석

이 문서는 `src/models/cb.py` 파일의 구조, 기능, 및 실행 로직에 대한 상세한 설명을 담고 있습니다.

## 1. 개요 (Overview)
이 스크립트는 **CatBoost** 모델을 위한 **단독 학습 및 최적화 모듈**입니다. `optimization.py`가 XGBoost/LGBM을 다뤘다면, 이 파일은 오직 CatBoost에 집중하여 특화된 최적화를 수행합니다. 특히 CatBoost의 강력한 기능인 **Ordered Boosting**과 **Native Categorical Feature Support**를 활용하여 최고 수준의 성능을 달성하는 것을 목표로 합니다.

## 2. 주요 설정 (Configuration)
- **DATA_DIR**: `data\03_resampled`
- **OUTPUT_DIR**: `data\05_optimized`
- **N_TRIALS**: `1` (실제 운영 시에는 값을 높여야 함, 예: 30~50)
- **전략**: 'Original' 데이터 사용 (SMOTE 미적용, `src/data/preprocess_and_split.py`의 결과물 활용).

## 3. 함수 별 기능 설명

### 3.1. `load_data()`
- `preprocess_and_split.py`에서 생성한 `X_train_original.csv`, `y_train_original.csv`, `X_test.csv`, `y_test.csv`를 로드합니다.
- **특이점**: CatBoost는 문자열 카테고리를 직접 처리할 수 있으나, 본 파이프라인에서는 앞선 전처리에 의해 이미 수치형으로 인코딩된 데이터를 사용할 가능성이 높습니다. (단, `optimization.py`와 달리 여기서는 명시적으로 원본 Raw 데이터를 다시 로드하는 로직은 없으므로, 인코딩된 상태의 데이터를 사용합니다.)

### 3.2. `find_optimal_threshold(y_true, y_prob)`
- `optimization.py`와 동일한 로직으로, F1-Score를 최대화하는 Cut-off 임계값을 탐색합니다.

### 3.3. `ModelOptimizer` 클래스
Optuna를 사용한 CatBoost 하이퍼파라미터 최적화 클래스입니다.

#### `objective(trial)`
- **파라미터 탐색 공간**:
  - `bagging_temperature`: 베이지안 부트스트랩의 강도를 조절.
  - `depth`: 트리의 깊이 (4~10).
  - `colsample_bylevel`: 레벨별 피처 샘플링 비율.
- **고정 파라미터 (CatBoost 특화)**:
  - `boosting_type`: `'Ordered'` (소규모 데이터에서 과적합 방지에 매우 효과적).
  - `bootstrap_type`: `'Bayesian'`.
  - `cat_features`: 범주형 변수 인덱스 리스트 전달.
  - `eval_metric`: `'F1'`.

### 3.4. `get_trained_model()`
- **역할**: 외부 모듈(예: `app.py`, `save_model.py`)에서 이 모델을 가져다 쓸 수 있도록 **학습된 모델 객체**를 반환하는 API 역할을 합니다.
- **실행 흐름**:
  1. **데이터 로드**: `load_data()` 호출.
  2. **범주형 변수 감지**: 데이터프레임에서 `object` 타입 컬럼을 자동 감지.
  3. **Optuna 최적화**: `ModelOptimizer`로 최적의 파라미터 탐색.
  4. **최종 모델 학습**: 찾은 Best Params로 전체 Train Set에 대해 재학습 (`iterations=1000`).
  5. **평가 및 리포트**: Test Set에 대한 Confusion Matrix, Classification Report, ROC-AUC 계산.
  6. **결과 저장**: `catboost_optimization_report.txt` 생성.
  7. **반환**: `final_model` 객체와 `X_train.columns` (피처 이름 리스트) 반환.

## 4. 메인 실행 흐름 (`main` 함수)
- `if __name__ == "__main__":` 블록은 이 파일이 직접 실행될 때 작동합니다.
- `get_trained_model()`을 호출하여 학습 프로세스 전체를 실행하고, 완료 로그를 출력합니다.
- 이 구조 덕분에 `python src/models/cb.py`로 단독 실행하여 모델을 만들 수도 있고, `import cb`로 다른 코드에서 모델 생성 함수만 빌려 쓸 수도 있습니다.

## 5. 생성되는 결과물 (data/05_optimized 폴더)
- `catboost_optimization_report.txt`: CatBoost 모델의 최적 파라미터와 최종 성능 지표가 기록된 텍스트 파일.

## 요약
이 파일은 CatBoost 모델의 생명주기(데이터 로드 -> 최적화 -> 학습 -> 평가 -> 배포 객체 생성)를 전담하는 핵심 모듈입니다. 특히 다른 모델들과 달리 `get_trained_model` 함수를 통해 **재사용성**을 고려하여 설계된 점이 특징입니다.
