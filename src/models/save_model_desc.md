# save_model.py 코드 상세 분석

이 문서는 `src/models/save_model.py` 파일의 구조, 기능, 및 실행 로직에 대한 상세한 설명을 담고 있습니다.

## 1. 개요 (Overview)
이 스크립트는 **최종 서비스 배포(Web App 등)**를 위해 학습된 모델 객체와 예측에 필요한 부가 정보들(Feature Names, 평균값 등)을 파일 형태로 저장하는 역할을 합니다. `cb.py` 모듈에 정의된 학습 로직을 가져와 실행하고, 결과물을 `churn_model.cbm`, `features.pkl`, `mean_values.pkl` 등의 파일로 직렬화하여 저장합니다.

## 2. 주요 설정 (Paths)
파일 저장 경로는 스크립트가 위치한 디렉토리(`src/models`)를 기준으로 절대 경로로 설정됩니다.
- `MODEL_PATH`: CatBoost 모델 파일 (`churn_model.cbm`)
- `FEATURES_PATH`: 학습에 사용된 피처 이름 리스트 (`features.pkl`)
- `MEAN_VALUES_PATH`: 이탈하지 않은 고객들의 피처 평균값 (`mean_values.pkl`) - 대시보드 등에서 비교 지표로 사용됨.

## 3. 함수 별 기능 설명

### 3.1. `save_model_and_features()`
이 함수는 다음과 같은 순차적인 작업을 수행합니다.

#### 1) 모델 학습 및 로드
```python
model, feature_names = get_trained_model()
```
- 외부 모듈 `cb.py`의 `get_trained_model()` 함수를 호출합니다.
- 이 과정에서 실제로 CatBoost 모델 학습이 수행되며, 학습 완료된 모델 객체와 피처 리스트를 반환받습니다.

#### 2) 모델 저장
```python
model.save_model(MODEL_PATH)
```
- CatBoost의 내장 메소드 `save_model`을 사용하여 모델을 전용 바이너리 포맷(`.cbm`)으로 저장합니다.

#### 3) 피처 리스트 저장
```python
pickle.dump(feature_names, f)
```
- Python의 `pickle` 모듈을 사용하여 피처 이름 리스트를 저장합니다.
- 이는 추후 입력 데이터의 컬럼 순서를 학습 시와 동일하게 맞추거나 검증할 때 필수적입니다.

#### 4) [중요] 평균값 데이터 생성 및 저장
```python
X_train, y_train, _, _, _ = load_and_split_data()
non_churn_X = X_train[y_train == 0]
mean_values = non_churn_X.mean(numeric_only=True).to_dict()
```
- **목적**: 웹 서비스나 대시보드에서 "평균적인 고객 대비 현재 고객의 상태"를 비교해서 보여주기 위함입니다.
- **로직**:
    - `load_and_split_data()`를 통해 학습 데이터를 다시 불러옵니다.
    - `y_train == 0` (이탈하지 않은 고객, 즉 충성 고객)인 데이터만 필터링합니다.
    - 수치형 변수들에 한해 **평균값(Mean)**을 계산하고 딕셔너리 형태로 변환합니다.
- **저장**: 계산된 딕셔너리를 `mean_values.pkl` 파일로 저장합니다.

## 4. 메인 실행 흐름 (`main` 함수)
- `if __name__ == "__main__":` 블록을 통해 스크립트가 직접 실행될 때만 `save_model_and_features()` 함수가 호출되도록 제어합니다.

## 5. 생성되는 결과물 (src/models 폴더 내)
스크립트 실행 시 다음 3가지 파일이 생성됩니다.
1.  **churn_model.cbm**: 학습 완료된 CatBoost 모델 파일.
2.  **features.pkl**: 모델 학습에 사용된 피처 이름 목록.
3.  **mean_values.pkl**: 잔존 고객들의 수치형 피처 평균값 데이터.

## 요약
이 파일은 모델링 단계에서 서비스 배포 단계로 넘어가는 연결 고리 역할을 합니다. 단순히 모델만 저장하는 것이 아니라, 실제 서비스 운영 시 사용자에게 유용한 인사이트(예: 평균 대비 나의 데이터 비교)를 제공하기 위한 부가 데이터(평균값)도 함께 생성하여 저장하는 것이 특징입니다.
