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
    # 전처리
    if 'international_plan' in df.columns:
        df['international_plan'] = (df['international_plan'] == 'yes').astype(int)
    if 'voice_mail_plan' in df.columns:
        df['voice_mail_plan'] = (df['voice_mail_plan'] == 'yes').astype(int)
    if 'churn' in df.columns:
        df['churn'] = (df['churn'] == 'yes').astype(int)
    return df

def ppt_visualizations():
    df = load_data()
    # 1. 🚨 VIP 전담 케어 (최우선 순위)
    # 4. 💰 요금 할인 쿠폰 발송 (가격 민감군)
    q2 = df['total_day_charge'].quantile(0.4)
    q3 = df['total_day_charge'].quantile(0.7)
    
    df['bill_group'] = pd.cut(
    df['total_day_charge'],
    bins=[0, q2, q3, 10000],
    labels=['~100%', '~90%', '~60%','~30%'],
    right=True, 
    )
    
    bill_df = df.groupby('bill_group', observed=False)['churn'].mean()

    bill_df.values
    x = bill_df.index       
    y = bill_df.values      
    color = ['lightgray','lightgray','firebrick']

    plt.bar(x, y, color=color, width=0.6)
    plt.xlabel('[국내] 낮 통신요금 상위(%)')
    plt.ylabel('이탈률')
    plt.title('Churn Rate by Day_bill_group')
    plt.show()

    # 2. 📞 불만 전담 마크 (고객 케어)
    dissatisfaction_df = df.groupby('number_customer_service_calls')['churn'].mean()

    dissatisfaction_df.values
    x = dissatisfaction_df.index         
    y = dissatisfaction_df.values        
    color = ['lightgray','lightgray','lightgray','lightgray',
            'orange', 'orange','firebrick','orange','orange','firebrick']

    plt.bar(x, y, color=color, width=0.6)
    plt.xlabel('Service Calls')
    plt.ylabel('Churn Rate')
    plt.title('Churn Rate by Customer Service Calls')
    plt.show()
    


    # 3. 🌍 국제전화 요금제 제안:
    df2 = df[df['international_plan']==1].copy()
     
    q1 = df2['total_intl_charge'].quantile(0.8)
    q2 = df2['total_intl_charge'].quantile(0.6)
    q3 = df2['total_intl_charge'].quantile(0.4)
    q4 = df2['total_intl_charge'].quantile(0.2)

    df2['intl_charge_group'] = pd.cut(
        df2['total_intl_charge'],
        bins=[0, q4, q3, q2, q1, 6],
        labels=['~100%', '~80%', '~60%','~40%','~20%'],
        right=True, 
    )

    intl_df = df.groupby('intl_charge_group', observed=False)['churn'].mean()

    x = intl_df.index       
    y = intl_df.values      
    color = ['lightgray','lightgray','lightgray','lightgray','firebrick']

    plt.bar(x, y, color=color, width=0.6)
    plt.xlabel('International Charge(%)')
    plt.ylabel('Churn Rate')
    plt.title('Churn Rate by intl_charge_group')
    plt.show()
   
        



