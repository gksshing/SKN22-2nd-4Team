import pandas as pd
import numpy as np
import optuna
import os
import sys
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, classification_report, precision_score, recall_score
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

# Configuration
DATA_DIR = r'c:\Workspaces\SKN22-2nd-4Team\data\03_resampled'
OUTPUT_DIR = r'c:\Workspaces\SKN22-2nd-4Team\data\05_optimized'
RANDOM_STATE = 42
N_TRIALS = 10 
DATASET_NAME = 'original'

def load_data(dataset_name):
    train_x_path = os.path.join(DATA_DIR, f"X_train_{dataset_name.lower()}.csv")
    train_y_path = os.path.join(DATA_DIR, f"y_train_{dataset_name.lower()}.csv")
    test_x_path = os.path.join(DATA_DIR, "X_test.csv")
    test_y_path = os.path.join(DATA_DIR, "y_test.csv")
    
    X_train = pd.read_csv(train_x_path)
    y_train = pd.read_csv(train_y_path).values.ravel()
    X_test = pd.read_csv(test_x_path)
    y_test = pd.read_csv(test_y_path).values.ravel()
    return X_train, y_train, X_test, y_test

def find_optimal_threshold(y_true, y_prob):
    """Finds the threshold that maximizes F1-score."""
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
    def __init__(self, X_train, y_train, model_name):
        self.X_train = X_train
        self.y_train = y_train
        self.model_name = model_name
        
        # Calculate scale_pos_weight for XGBoost (Neg/Pos ratio)
        # Using simple integer counts
        n_pos = sum(y_train)
        n_neg = len(y_train) - n_pos
        self.scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
    def objective(self, trial):
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        
        if self.model_name == 'XGBoost':
            params = {
                'booster': 'dart', # Fixed
                'grow_policy': 'depthwise', # Fixed
                'n_estimators': 200,
                'eval_metric': 'logloss', # Fixed
                'random_state': RANDOM_STATE,
                'scale_pos_weight': self.scale_pos_weight, # Handle Imbalance
                'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'n_jobs': -1
            }
            model = XGBClassifier(**params)
            
        elif self.model_name == 'LightGBM':
            params = {
                'boosting_type': 'dart', # Fixed
                'n_estimators': 200,
                'random_state': RANDOM_STATE,
                'verbose': -1,
                'class_weight': 'balanced', # Handle Imbalance
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'n_jobs': -1
            }
            model = LGBMClassifier(**params)
            
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        # Optuna pruning integration with Cross Validation
        # Since standard cross_val_score doesn't support pruning easily, 
        # we will use a simpler loop or just standard CV for F1.
        # For simplicity and robustness with F1, we use cross_val_score.
        # Pruning inside CV is complex without a manual loop. 
        # We will maximize F1.
        
        scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1')
        return scores.mean()

def run_optimization():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading Dataset: {DATASET_NAME}")
    X_train, y_train, X_test, y_test = load_data(DATASET_NAME)
    
    # Calc ratio for final model
    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    models_to_tune = ['XGBoost','LightGBM']
    best_results = {}
    
    for model_name in models_to_tune:
        print(f"\n{'='*40}")
        print(f"Optimizing {model_name}...")
        print(f"{'='*40}")
        
        optimizer = ModelOptimizer(X_train, y_train, model_name)
        
        study = optuna.create_study(
            direction='maximize', 
            sampler=TPESampler(seed=RANDOM_STATE),
            pruner=HyperbandPruner()
        )
        
        study.optimize(optimizer.objective, n_trials=N_TRIALS, show_progress_bar=True)
        
        print("\nOptimization Finished.")
        print(f"Best Trial F1: {study.best_value:.4f}")
        print("Best Params:", study.best_params)
        
        # --- Final Evaluation logic ---
        print(f"\nTraining Final {model_name} with Best Params...")
        
        best_params = study.best_params
        
        # Re-instantiate with Best Params + Fixed Params needed for training
        if model_name == 'XGBoost':
            final_model = XGBClassifier(
                booster='dart', grow_policy='depthwise', n_estimators=1000, eval_metric='logloss',
                random_state=RANDOM_STATE, use_label_encoder=False, scale_pos_weight=scale_pos_weight, **best_params
            )
        elif model_name == 'LightGBM':
            final_model = LGBMClassifier(
                boosting_type='dart', n_estimators=1000, random_state=RANDOM_STATE, verbose=-1, class_weight='balanced', **best_params
            )
            
        final_model.fit(X_train, y_train)
        
        # Predict Probabilities
        y_prob = final_model.predict_proba(X_test)[:, 1]
        
        # --- Threshold Tuning ---
        print("Optimizing Prediction Threshold...")
        best_thresh, best_f1_test = find_optimal_threshold(y_test, y_prob)
        
        # Apply Threshold
        y_pred_opt = (y_prob >= best_thresh).astype(int)
        
        # Final Metrics
        roc_auc = roc_auc_score(y_test, y_prob)
        print("\n--- Final Test Report (Optimized Threshold) ---")
        print(f"Best Threshold: {best_thresh:.2f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(classification_report(y_test, y_pred_opt))
        
        # Save Report
        result_text = (
            f"Model: {model_name}\n"
            f"Dataset: {DATASET_NAME}\n"
            f"Best Params: {best_params}\n"
            f"Best Validation F1: {study.best_value}\n"
            f"Test ROC AUC: {roc_auc}\n"
            f"Optimal Threshold: {best_thresh}\n"
            f"Test F1 (Optimized): {best_f1_test}\n\n"
            f"Classification Report:\n{classification_report(y_test, y_pred_opt)}"
        )
        
        save_path = os.path.join(OUTPUT_DIR, f"{model_name.lower()}_optimization_report.txt")
        with open(save_path, "w") as f:
            f.write(result_text)
        print(f"Results saved to {save_path}")

if __name__ == "__main__":
    run_optimization()
