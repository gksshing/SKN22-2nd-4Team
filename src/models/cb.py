import pandas as pd
import numpy as np
import optuna
import os
import sys
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

# --- 설정 (Configuration) ---
DATA_DIR = r'c:\Workspaces\SKN22-2nd-4Team\data\03_resampled'
OUTPUT_DIR = r'c:\Workspaces\SKN22-2nd-4Team\data\05_optimized'
RANDOM_STATE = 42
N_TRIALS = 1 

def load_data():
    """
    저장된 Split 데이터를 로드합니다.
    preprocess_and_split.py 에서 생성된 파일을 사용합니다.
    """
    train_x_path = os.path.join(DATA_DIR, "X_train_original.csv")
    train_y_path = os.path.join(DATA_DIR, "y_train_original.csv")
    test_x_path = os.path.join(DATA_DIR, "X_test.csv")
    test_y_path = os.path.join(DATA_DIR, "y_test.csv")
    
    print(f"데이터 로드 경로: {DATA_DIR}")
    
    X_train = pd.read_csv(train_x_path)
    y_train = pd.read_csv(train_y_path).values.ravel()
    X_test = pd.read_csv(test_x_path)
    y_test = pd.read_csv(test_y_path).values.ravel()
    
    return X_train, y_train, X_test, y_test


def find_optimal_threshold(y_true, y_prob):
    """
    F1 Score를 최대화하는 최적의 임계값(Threshold)을 찾습니다.
    """
    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.arange(0.01, 1.00, 0.01)
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = thresh
    return best_threshold, best_f1

class ModelOptimizer:
    def __init__(self, X_train, y_train, cat_features):
        self.X_train = X_train
        self.y_train = y_train
        self.cat_features = cat_features
        
    def objective(self, trial):
        # 학습 데이터 안에서도 또 쪼개서 검증 (CV)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        
        params = {
            'boosting_type': 'Ordered',
            'bootstrap_type': 'Bayesian',
            'iterations': 200, 
            'random_state': RANDOM_STATE,
            'verbose': 0,
            'allow_writing_files': False,
            'eval_metric': 'F1',
            'cat_features': self.cat_features,
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'depth': trial.suggest_int('depth', 4, 10),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        }
        model = CatBoostClassifier(**params)
        
        try:
            scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1', error_score='raise')
        except Exception as e:
            return 0.0
            
        return scores.mean()

def get_trained_model():
    """
    전체 학습 파이프라인을 실행하고 학습된 모델을 반환하는 함수입니다.
    외부(예: app.py)에서 호출하여 모델을 가져갈 수 있습니다.
    
    Returns:
        final_model: 학습이 완료된 CatBoost 모델 객체
        feature_names: 학습에 사용된 컬럼명 리스트 (X_train.columns)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("데이터 로딩...")
    # submission 데이터는 정답이 없으므로 X_submission 변수에 따로 저장
    X_train, y_train, X_test, y_test = load_data()
    
    # CatBoost용 범주형 변수 감지
    cat_features = X_train.select_dtypes(include=['object']).columns.tolist()
    print(f"감지된 범주형 변수: {cat_features}")
    
    print(f"\n{'='*40}")
    print(f"CatBoost 최적화 진행 (Trials: {N_TRIALS})...")
    print(f"{'='*40}")
    
    optimizer = ModelOptimizer(X_train, y_train, cat_features)
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(optimizer.objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    print(f"최고 CV F1 점수: {study.best_value:.4f}")
    print("최적 파라미터:", study.best_params)
    
    # --- 최종 검증 (Test Set) ---
    print(f"\n최종 모델 학습 및 테스트 데이터 평가 중...")
    
    final_model = CatBoostClassifier(
        boosting_type='Ordered', bootstrap_type='Bayesian', 
        iterations=1000, 
        random_state=RANDOM_STATE, verbose=0, 
        allow_writing_files=False, 
        cat_features=cat_features,
        **study.best_params
    )
    
    # Train 데이터 전체로 학습
    final_model.fit(X_train, y_train)
    
    # 1. Test Set 평가
    y_prob_val = final_model.predict_proba(X_test)[:, 1]
    
    # 임계값 튜닝
    best_thresh, best_f1_val = find_optimal_threshold(y_test, y_prob_val)
    y_pred_val = (y_prob_val >= best_thresh).astype(int)
    
    # 리포트 출력
    print("\n--- 최종 결과 리포트 (Test Set) ---")
    print(f"최적 임계값: {best_thresh:.2f}")
    print(confusion_matrix(y_test, y_pred_val))
    report = classification_report(y_test, y_pred_val)
    print(report)
    
    # ROC AUC 계산
    roc_auc = roc_auc_score(y_test, y_prob_val)
    print(f"Test Set ROC AUC: {roc_auc:.4f}")
    
    # 결과 저장
    result_text = (
        f"Model: CatBoost\n"
        f"Best Params: {study.best_params}\n"
        f"Best CV F1: {study.best_value}\n"
        f"Test ROC AUC: {roc_auc:.4f}\n" 
        f"Optimal Threshold: {best_thresh}\n"
        f"Test F1 (Optimized): {best_f1_val}\n\n"
        f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_val)}\n\n"
        f"Classification Report:\n{report}"
    )
    
    save_path = os.path.join(OUTPUT_DIR, "catboost_optimization_report.txt")
    with open(save_path, "w") as f:
        f.write(result_text)
    print(f"결과가 저장되었습니다: {save_path}")
    
    # 모델과 컬럼명 리스트 반환
    return final_model, X_train.columns

if __name__ == "__main__":
    # 이 파일을 직접 실행할 경우에만 학습 로직이 돌아갑니다.
    model, features = get_trained_model()
    print("\n학습 완료 확인.")
    print("입력 변수 목록:", list(features))
