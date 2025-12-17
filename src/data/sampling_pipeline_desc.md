# sampling_pipeline.py 코드 상세 분석

이 문서는 `src/data/sampling_pipeline.py` 파일의 구조, 기능, 및 실행 로직에 대한 상세한 설명을 담고 있습니다.

## 1. 개요 (Overview)
이 스크립트는 불균형한 데이터셋 문제를 해결하기 위해 **다양한 오버샘플링(Over-sampling) 및 언더샘플링(Under-sampling) 기법**을 적용하여 여러 버전의 학습 데이터(Train Set)를 생성하는 파이프라인입니다. `preprocess_and_split.py`와 유사하게 전처리와 데이터 분할을 수행하지만, 그 후 **SMOTE, SMOTE-Tomek, SMOTE-ENN** 알고리즘을 추가로 적용하여 리샘플링된 데이터를 저장합니다.

## 2. 주요 라이브러리 및 의존성
- **imblearn**: 불균형 데이터 처리를 위한 핵심 라이브러리.
  - `SMOTE`: 소수 클래스 데이터를 합성하여 늘리는 기법.
  - `SMOTETomek`, `SMOTEENN`: SMOTE에 각각 Tomek Links, ENN(Edited Nearest Neighbours) 언더샘플링을 결합한 복합 샘플링 기법.
- **preprocess_and_split**: 이전 모듈에서 정의한 `load_data`, `preprocess_data` 함수 등을 재사용하여 일관된 전처리를 보장합니다.

## 3. 함수 별 기능 설명

### 3.1. `get_sampling_strategies()`
- **기능**: 적용할 샘플링 전략들을 딕셔너리 형태로 정의하여 반환합니다.
- **포함된 전략**:
  1. `SMOTE`: 기본 SMOTE 알고리즘.
  2. `SMOTE_Tomek`: 오버샘플링 후 경계선 관측치 정리 (복합).
  3. `SMOTE_ENN`: 오버샘플링 후 노이즈 관측치 정리 (복합).
- **공통 설정**: 모든 알고리즘에 `random_state=42`를 적용하여 결과의 재현성을 확보합니다.

### 3.2. `print_class_distribution(y, name="Dataset")`
- **기능**: 현재 데이터의 클래스 분포(개수 및 비율)를 출력하여 불균형 정도나 샘플링 효과를 시각적으로 확인시켜줍니다.

### 3.3. `save_resampled_data(X_res, y_res, method_name)`
- **기능**: 리샘플링된 데이터를 CSV 파일로 저장합니다.
- **경로**: `c:\Workspaces\SKN22-2nd-4Team\data\03_resampled`
- **파일명 규칙**: `X_train_{method_name}.csv`, `y_train_{method_name}.csv` (소문자 변환).

## 4. 메인 실행 흐름 (`main` 함수)

1.  **데이터 로드 및 전처리 (Load & Preprocess)**:
    - `preprocess_and_split.py`의 로직을 그대로 재사용하여 데이터를 로드하고 전처리합니다.
    - 이를 통해 모든 실험에서 동일한 베이스 데이터가 사용됨을 보장합니다.

2.  **데이터 분할 (Split)**:
    - 학습용(`train`)과 테스트용(`test`) 데이터로 분리합니다.
    - **중요**: 샘플링(Sampling)은 **오직 학습 데이터(Train Set)에만 적용**해야 합니다. 테스트 데이터(Test Set)는 원본 분포를 유지해야 올바른 평가가 가능하기 때문입니다.

3.  **샘플링 전략 적용 (Apply Sampling)**:
    - `get_sampling_strategies()`에서 정의한 3가지 전략을 순회하며 실행합니다.
    - 각 전략별로:
        - `fit_resample`을 통해 새로운 `X_res`, `y_res`를 생성합니다.
        - `print_class_distribution`을 통해 샘플링 후 클래스 비율 변화를 출력합니다 (보통 50:50 근처로 맞춰짐).
        - 결과를 파일로 저장합니다.

4.  **원본 데이터 저장 (Save Original)**:
    - 비교 실험을 위해 샘플링을 적용하지 않은 **원본 학습 데이터(`original`)**도 별도로 저장합니다.
    - 변하지 않아야 할 **테스트 데이터(`test`)**도 함께 저장합니다.

## 5. 생성되는 파일 목록
이 스크립트를 실행하면 `data\03_resampled` 폴더에 다음과 같은 파일 세트들이 생성됩니다:

1.  **Original (기준)**: `X_train_original.csv`, `y_train_original.csv`
2.  **SMOTE**: `X_train_smote.csv`, `y_train_smote.csv`
3.  **SMOTE-Tomek**: `X_train_smote_tomek.csv`, `y_train_smote_tomek.csv`
4.  **SMOTE-ENN**: `X_train_smote_enn.csv`, `y_train_smote_enn.csv`
5.  **Test Set**: `X_test.csv`, `y_test.csv` (공통 평가용)

## 요약
이 파일은 다양한 불균형 처리 기법을 실험하기 위한 데이터 준비 스크립트입니다. 데이터 누수(Leakage)를 방지하기 위해 데이터를 먼저 분할한 후 학습 데이터에만 샘플링을 적용하는 모범 사례를 따르고 있으며, 여러 파생 데이터셋을 자동 생성하여 후속 모델링 단계에서 비교 실험을 용이하게 합니다.
