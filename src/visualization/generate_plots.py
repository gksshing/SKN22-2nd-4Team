import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pickle

# 설정
DATA_PATH = "data/01_raw/train.csv"
MODEL_PATH = "src/models/churn_model.cbm"
OUTPUT_DIR = "presentation_assets"

# 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Preprocessing
    if 'international_plan' in df.columns:
        df['international_plan'] = (df['international_plan'] == 'yes').astype(int)
    if 'voice_mail_plan' in df.columns:
        df['voice_mail_plan'] = (df['voice_mail_plan'] == 'yes').astype(int)
    return df

def generate_visualizations():
    df = load_data()
    
    # 1. Class Distribution (Before Sampling)
    plt.figure(figsize=(8, 6))
    sns.countplot(x='churn', data=df, palette='viridis')
    plt.title('클래스 분포 (불균형 데이터)')
    plt.xlabel('Churn (0: 유지, 1: 이탈)')
    plt.ylabel('고객 수')
    plt.savefig(os.path.join(OUTPUT_DIR, '01_class_distribution.png'))
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    # Select numeric features only + target
    numeric_df = df.select_dtypes(include=['number'])
    # Add churn if it's 'yes'/'no' -> convert first
    if df['churn'].dtype == 'object':
         numeric_df['churn'] = (df['churn'] == 'yes').astype(int)
    else:
         numeric_df['churn'] = df['churn']
         
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title('주요 변수 간 상관관계')
    plt.savefig(os.path.join(OUTPUT_DIR, '02_correlation_heatmap.png'))
    plt.close()

    # Model Evaluation Plots
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    
    # Prepare X, y
    feature_names = model.feature_names_
    try:
        X = df[feature_names]
        # Target needs to be numeric
        y = (df['churn'] == 'yes').astype(int) if df['churn'].dtype == 'object' else df['churn']
        
        # Split for evaluation demo (8:2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 3. Feature Importance
        importances = model.get_feature_importance()
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(10)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=fi_df, palette='magma')
        plt.title('Top 10 중요 변수 (Feature Importance)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '03_feature_importance.png'))
        plt.close()
        
        # 4. Confusion Matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Test Set)')
        plt.savefig(os.path.join(OUTPUT_DIR, '04_confusion_matrix.png'))
        plt.close()
        
        # 5. ROC Curve
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(OUTPUT_DIR, '05_roc_curve.png'))
        plt.close()

    except Exception as e:
        print(f"Error generating model plots: {e}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    generate_visualizations()
    print("Visualizations generated successfully.")
