import os
import pickle
import pandas as pd
from catboost import CatBoostClassifier
from cb import get_trained_model, load_data

# 현재 스크립트의 절대 경로를 기준으로 저장 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "churn_model.cbm")
FEATURES_PATH = os.path.join(CURRENT_DIR, "features.pkl")
MEAN_VALUES_PATH = os.path.join(CURRENT_DIR, "mean_values.pkl")

def save_model_and_features():
    print("모델 학습을 시작합니다...")
    # cb.py의 get_trained_model 함수를 호출하여 학습된 모델과 피처 이름을 가져옴
    model, feature_names = get_trained_model()
    
    print(f"모델 학습 완료. 모델을 저장합니다: {MODEL_PATH}")
    # CatBoost 모델 저장
    model.save_model(MODEL_PATH)
    
    print(f"피처 리스트를 저장합니다: {FEATURES_PATH}")
    # Feature names 리스트 저장 (pickle 사용)
    with open(FEATURES_PATH, 'wb') as f:
        pickle.dump(feature_names, f)
        
    # --- 추가: 평균값 계산 (이탈하지 않은 고객 기준) ---
    print("평균값(이탈 안 한 고객)을 계산합니다...")
    X_train, y_train, _, _ = load_data()
    
    # 이탈하지 않은 고객(0)만 필터링
    # y_train이 Series라면 인덱스로 매칭
    non_churn_X = X_train[y_train == 0]
    
    # 수치형 변수들의 평균 계산
    # (범주형 변수는 평균이 의미 없으므로 제외하거나, CatBoost가 처리하는 방식에 따라 다름)
    # 여기서는 간단히 numeric_only=True로 계산
    mean_values = non_churn_X.mean(numeric_only=True).to_dict()
    
    print(f"평균값 데이터를 저장합니다: {MEAN_VALUES_PATH}")
    with open(MEAN_VALUES_PATH, 'wb') as f:
        pickle.dump(mean_values, f)
        
    print("모든 저장이 완료되었습니다.")

if __name__ == "__main__":
    save_model_and_features()
