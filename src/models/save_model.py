import os
import pickle
from catboost import CatBoostClassifier
from cb import get_trained_model

# 현재 스크립트의 절대 경로를 기준으로 저장 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "churn_model.cbm")
FEATURES_PATH = os.path.join(CURRENT_DIR, "features.pkl")

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
        
    print("모든 저장이 완료되었습니다.")

if __name__ == "__main__":
    save_model_and_features()
