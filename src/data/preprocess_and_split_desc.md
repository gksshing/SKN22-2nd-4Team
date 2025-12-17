# preprocess_and_split.py 코드 상세 분석

이 문서는 `src/data/preprocess_and_split.py` 파일의 구조, 기능, 및 실행 로직에 대한 상세한 설명을 담고 있습니다.

## 1. 개요 (Overview)
이 스크립트는 원본 데이터(`train.csv`, `test.csv`)를 로드하여 전처리 과정을 거친 후, 기계 학습 모델 학습을 위해 **학습용 데이터(Train Set)**와 **테스트용 데이터(Test Set)**로 분할하여 저장하는 역할을 수행합니다. 클래스 불균형 문제를 고려하여 `Stratified Split`(층화 추출)을 사용합니다.

## 2. 주요 설정 (Configuration)
코드 상단에 정의된 상수는 데이터 경로와 핵심 변수를 제어합니다.

- `RAW_DATA_PATH`: 원본 데이터가 위치한 경로 (`data\01_raw`).
- `TRAIN_FILE` / `TEST_FILE`: 로드할 파일명.
- `TARGET_COL`: 예측할 타겟 변수명 (`churn`).
- `RANDOM_STATE`: 재현성을 위한 난수 시드값 (42).

## 3. 함수 별 기능 설명

### 3.1. `load_data()`
- **기능**: `train.csv`와 `test.csv`를 로드하고 하나로 병합합니다.
- **로직**:
    1. 각 파일을 Pandas DataFrame으로 읽어옵니다.
    2. `test.csv`에 타겟 컬럼(`churn`)이 있는지 확인합니다. 만약 없다면, 나중에 분리하거나 무시하기 위해 `NaN` 값으로 생성합니다.
    3. 데이터의 출처를 구분하기 위해 `dataset_source` 컬럼('train' 또는 'test')을 추가합니다.
    4. 두 데이터 프레임을 하나로 병합(`concat`)하여 반환합니다.

### 3.2. `preprocess_data(df)`
- **기능**: 범주형 데이터(Categorical Data)를 수치형(Numerical)으로 변환합니다.
- **상세 처리 과정**:
    1. **이진 매핑 (Binary Mapping)**: 
        - `international_plan`, `voice_mail_plan` 컬럼의 값 ('yes', 'no')을 (1, 0)으로 변환합니다.
        - 타겟 변수 `churn`도 동일하게 (1, 0)으로 변환합니다.
    2. **라벨 인코딩 (Label Encoding)**:
        - `state` (주), `area_code` (지역 코드) 컬럼을 `sklearn.preprocessing.LabelEncoder`를 사용하여 고유한 정수 값으로 변환합니다.

### 3.3. `remove_outliers_iqr(df, columns, factor=1.5)`
- **기능**: IQR (Interquartile Range) 방식을 사용하여 이상치를 제거합니다.
- **로직**:
    - 지정된 각 컬럼에 대해 Q1(25%)과 Q3(75%)를 계산합니다.
    - `IQR = Q3 - Q1`을 계산하고, `(Q1 - 1.5 * IQR)` 미만이거나 `(Q3 + 1.5 * IQR)` 초과인 데이터를 필터링하여 제거합니다.
- **참고**: 현재 `main()` 함수 설정 상 **비활성화**되어 있습니다. (데이터 손실 방지 목적일 가능성).

## 4. 메인 실행 흐름 (`main` 함수)

`if __name__ == "__main__":` 블록에서 실행되는 전체 프로세스는 다음과 같습니다.

1.  **데이터 로드**: `load_data()` 호출.
2.  **전처리**: `preprocess_data()` 호출.
3.  **이상치 제거 플래그 확인**:
    - `APPLY_OUTLIER_REMOVAL = False`로 설정되어 있어 이상치 제거는 수행되지 않고 넘어갑니다.
4.  **분할 준비 (Missing Values Handling)**:
    - 학습/테스트 분할(Stratified Split)을 수행하기 위해 **타겟 값(`churn`)이 없는 행들을 제거**합니다. (주로 라벨이 없는 `test.csv` 데이터가 제거됨).
    - 불필요한 컬럼 (`churn`, `dataset_source`, `id`)을 제거하여 Feature Matrix `X`와 Target Vector `y`를 생성합니다.
5.  **데이터 분할 (Splitting)**:
    - 데이터 크기에 따라 분할 비율을 유동적으로 결정합니다.
        - 전체 데이터가 5000개인 경우: Test Set을 **750개**로 고정.
        - 그 외의 경우 (예: 4250개): Test Set 비율을 **15%**로 설정.
    - `train_test_split` 함수를 사용하며, `stratify=y` 옵션을 켜서 타겟 클래스 비율을 유지합니다.
6.  **검증 (Verification)**:
    - 분할된 `X_train`, `X_test`, `y_train`, `y_test`의 크기(Shape)를 출력하여 의도한 대로 분할되었는지 확인합니다.
    - 의도한 크기(4250, 750)와 다를 경우 경고 메시지를 출력합니다.
7.  **데이터 저장**:
    - 결과 파일들을 `c:\Workspaces\SKN22-2nd-4Team\data\03_resampled` 경로에 저장합니다.
    - 저장되는 파일 목록:
        - `X_train_original.csv`: 학습용 피처 데이터
        - `X_test.csv`: 테스트용 피처 데이터
        - `y_train_original.csv`: 학습용 타겟 데이터
        - `y_test.csv`: 테스트용 타겟 데이터

## 요약
이 파일은 원본 데이터를 읽어 머신러닝 모델이 학습할 수 있는 형태(수치형 변환 완료)로 가공하고, 층화 추출을 통해 신뢰성 있는 학습/평가 데이터셋을 생성하여 저장하는 파이프라인의 핵심 전처리 스크립트입니다.
