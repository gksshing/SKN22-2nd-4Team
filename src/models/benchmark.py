import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Configuration
DATA_DIR = r'c:\Workspaces\SKN22-2nd-4Team\data\03_resampled'
OUTPUT_DIR = r'c:\Workspaces\SKN22-2nd-4Team\data\04_results'
RANDOM_STATE = 42

def load_dataset(dataset_name):
    """Loads X_train, y_train for a specific sampling strategy, and the common Test set."""
    # Construct filenames based on convention: X_train_smote.csv
    train_x_path = os.path.join(DATA_DIR, f"X_train_{dataset_name.lower()}.csv")
    train_y_path = os.path.join(DATA_DIR, f"y_train_{dataset_name.lower()}.csv")
    
    test_x_path = os.path.join(DATA_DIR, "X_test.csv")
    test_y_path = os.path.join(DATA_DIR, "y_test.csv")
    
    if not os.path.exists(train_x_path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {train_x_path}")
        
    X_train = pd.read_csv(train_x_path)
    y_train = pd.read_csv(train_y_path).values.ravel() # Ensure 1D array
    X_test = pd.read_csv(test_x_path)
    y_test = pd.read_csv(test_y_path).values.ravel()
    
    return X_train, y_train, X_test, y_test

def get_models(use_class_weights=False):
    """Returns a dict of models. SVM, LR, ANN are wrapped in Pipelines with Scaling."""
    
    models = {}
    
    # Weights configuration
    # Note: XGBoost uses scale_pos_weight = count(negative) / count(positive)
    # But for simplicity in this loop, we might need manual calc if library doesn't support 'balanced' string.
    # LightGBM/CatBoost/RF/SVC/LR support 'balanced'.
    
    # Simple ratio estimate for XGBoost/CatBoost if needed (approx 850 non-churn / 150 churn ~ 5.6)
    # But let's stick to 'balanced' string where supported, or manual logic.
    
    cw_param = 'balanced' if use_class_weights else None
    
    # 1. Tree/Ensemble Models
    # DT
    models['DT'] = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight=cw_param)
    
    # RF
    models['RF'] = RandomForestClassifier(random_state=RANDOM_STATE, class_weight=cw_param)
    
    # XGBoost (doesn't support 'balanced' string directly usually, needs scale_pos_weight)
    # We will compute a rough scale_pos_weight if needed, or leave default if not strict.
    # For this benchmark, let's use a fixed weight ~5.7 (Total 3333 / 483 churn ~ 14.5% churn. Neg/Pos = 2850/483 ~ 5.9)
    xgb_weight = 5.9 if use_class_weights else 1
    models['XGBoost'] = XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE, use_label_encoder=False, scale_pos_weight=xgb_weight)
    
    # LightGBM
    models['LightGBM'] = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, class_weight=cw_param)
    
    # CatBoost (benchmark generic)
    # CatBoost supports auto_class_weights='Balanced'
    cb_weights = 'Balanced' if use_class_weights else None
    models['CatBoost'] = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE, auto_class_weights=cb_weights)
    
    # 2. Distance/Gradient-based Models (Need Scaling)
    # ANN (MLP doesn't strictly support class_weight in sklearn < 0.24 or so? It does not have class_weight param usually)
    # We skip weight for ANN or leave as is.
    models['ANN'] = make_pipeline(StandardScaler(), MLPClassifier(random_state=RANDOM_STATE, max_iter=1000))
    
    # SVM
    models['SVM'] = make_pipeline(StandardScaler(), SVC(probability=True, random_state=RANDOM_STATE, class_weight=cw_param))
    
    # LR
    models['LR'] = make_pipeline(StandardScaler(), LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight=cw_param))
    
    return models

def load_raw_data_for_catboost():
    """
    Loads raw data and splits it exactly as the benchmark datasets were likely created
    (random_state=42), preserving categorical features for CatBoost.
    """
    raw_path = os.path.join(r'c:\Workspaces\SKN22-2nd-4Team\data\01_raw', "train.csv")
    df = pd.read_csv(raw_path)
    
    # Simple Preprocessing (Yes/No -> 1/0)
    for col in ['international_plan', 'voice_mail_plan']:
        if col in df.columns:
            df[col] = (df[col] == 'yes').astype(int)
    
    # Target
    X = df.drop('churn', axis=1)
    y = df['churn'].apply(lambda x: 1 if x == 'yes' else 0)
    
    # Split (Must match the split used for X_train_original.csv generation)
    # Assuming standard 80/20 split with seed 42 was used.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    return X_train, y_train, X_test, y_test, cat_features

def evaluate_models_on_dataset(dataset_name):
    """
    Train and evaluate all models on a specific dataset (e.g., 'smote').
    Returns: Results DataFrame, and a list of ROC data dictionaries.
    """
    print(f"\n[{dataset_name.upper()}] Loading data...")
    try:
        X_train, y_train, X_test, y_test = load_dataset(dataset_name)
    except FileNotFoundError as e:
        print(e)
        return None, None

    # Apply class weights ONLY for original dataset (as requested to replace SMOTE)
    use_weights = (dataset_name == 'original')
    models = get_models(use_class_weights=use_weights)
    
    results = []
    roc_data = {} # store per model
    
    print(f"[{dataset_name.upper()}] Training {len(models)} models (Class Weights={use_weights})...")
    
    for name, model in models.items():
        # Special handling for CatBoost on Original dataset
        # to use Native Categorical Support (as requested by User)
        if name == 'CatBoost' and dataset_name == 'original':
            print("  > CatBoost: Switching to Raw Data for Native Categorical Handling...")
            X_train_cb, y_train_cb, X_test_cb, y_test_cb, cat_features = load_raw_data_for_catboost()
            
            # Re-initialize with cat_features AND Class Weights
            model = CatBoostClassifier(
                verbose=0, 
                random_state=RANDOM_STATE,
                cat_features=cat_features,
                auto_class_weights='Balanced'
            )
            model.fit(X_train_cb, y_train_cb)
            
            y_pred = model.predict(X_test_cb)
            y_prob = model.predict_proba(X_test_cb)[:, 1]
            
            # Metrics (using the CB specific test set, which should be identical in labels)
            p = precision_score(y_test_cb, y_pred, zero_division=0)
            r = recall_score(y_test_cb, y_pred, zero_division=0)
            f1 = f1_score(y_test_cb, y_pred, zero_division=0)
            roc = roc_auc_score(y_test_cb, y_prob)
            
            # ROC Data for plotting (Use y_test_cb)
            fpr, tpr, _ = roc_curve(y_test_cb, y_prob)
            roc_data[name] = (fpr, tpr, roc)
            
        else:
            # Standard Path
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            p = precision_score(y_test, y_pred, zero_division=0)
            r = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc = roc_auc_score(y_test, y_prob)
            
            # ROC Data for plotting (Use standard y_test)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data[name] = (fpr, tpr, roc)
        
        results.append({
            'Model': name,
            'Precision': p,
            'Recall': r,
            'F1-score': f1,
            'ROC AUC': roc
        })
        
        print(f"  > {name}: F1={f1:.4f}, AUC={roc:.4f}")
        
    df_results = pd.DataFrame(results).set_index('Model')
    return df_results, roc_data

def plot_roc_curves(roc_data, dataset_name):
    """Plots and saves ROC curves for all models in one figure."""
    plt.figure(figsize=(10, 8))
    
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {dataset_name.upper()} Dataset')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    save_path = os.path.join(OUTPUT_DIR, f"roc_curve_{dataset_name.lower()}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  ROC Plot saved to: {save_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 4 Dataset Variants from previous step
    datasets = ['original', 'smote', 'smote_tomek', 'smote_enn']
    
    all_benchmarks_completed = True
    
    for ds_name in datasets:
        print(f"\n{'='*40}")
        print(f"Processing Dataset: {ds_name}")
        print(f"{'='*40}")
        
        results_df, roc_data = evaluate_models_on_dataset(ds_name)
        
        if results_df is not None:
            # Save Table
            csv_path = os.path.join(OUTPUT_DIR, f"benchmark_{ds_name.lower()}.csv")
            results_df.to_csv(csv_path)
            print(f"  Table saved to: {csv_path}")
            
            # Print for logs
            print(results_df)
            
            # Plot ROC
            plot_roc_curves(roc_data, ds_name)
        else:
            all_benchmarks_completed = False

    if all_benchmarks_completed:
        print("\nAll benchmarks completed successfully.")
    else:
        print("\nSome benchmarks failed due to missing files.")

if __name__ == "__main__":
    main()
