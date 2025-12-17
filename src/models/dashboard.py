import streamlit as st
import pandas as pd
import pickle
import os
from catboost import CatBoostClassifier
import plotly.express as px
import plotly.graph_objects as go

# --- 1. 설정 및 데이터 로드 ---
st.set_page_config(page_title="고객 이탈 관리 대시보드", layout="wide", page_icon="📊")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "../../data/01_raw/train.csv") # 상대 경로 주의
MODEL_PATH = os.path.join(CURRENT_DIR, "churn_model.cbm")
FEATURE_PATH = os.path.join(CURRENT_DIR, "features.pkl")

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
        return None
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return None, None
    
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    
    with open(FEATURE_PATH, 'rb') as f:
        feature_names = pickle.load(f)
        
    return model, feature_names

# 데이터 및 모델 로드
df = load_data()
model, feature_names = load_model()

if df is not None and model is not None:
    
    # --- 2. 배치 예측 및 파생 변수 생성 ---
    
    # 1. 전처리 (Global Preprocessing) - df에 바로 적용하여 모델 입력 및 전략 로직 모두 해결
    if 'international_plan' in df.columns and df['international_plan'].dtype == 'object':
        df['international_plan'] = (df['international_plan'] == 'yes').astype(int)
    if 'voice_mail_plan' in df.columns and df['voice_mail_plan'].dtype == 'object':
        df['voice_mail_plan'] = (df['voice_mail_plan'] == 'yes').astype(int)
        
    # 모델 입력용 데이터 준비 (feature_names 순서 맞춤)
    # 없는 컬럼은 0으로 채우거나 처리해야 하지만, train.csv가 원본이라 가정하고 필요한 컬럼만 추출
    # 실제로는 전처리가 필요할 수 있으나, 여기서는 간단히 raw data 사용 (CatBoost라 범주형 처리 자동)
    # 단, 학습때 사용한 컬럼만 선택
    try:
        X = df[feature_names]
    except KeyError as e:
        st.error(f"데이터에 필요한 컬럼이 없습니다: {e}")
        st.stop()

    # 배치 예측 (이탈 확률)
    # CatBoost predict_proba -> [class0_prob, class1_prob]

    # 전처리 완료됨 (위에서 처리)
        
    probs = model.predict_proba(X)[:, 1]
    df['Probability'] = probs
    
    # 월 총 요금 (Total Bill)
    df['total_bill'] = (
        df['total_day_charge'] + 
        df['total_eve_charge'] + 
        df['total_night_charge'] + 
        df['total_intl_charge']
    )
    
    # 기대 손실액 (Revenue at Risk)
    df['risk_value'] = df['total_bill'] * df['Probability']
    
    # 위험 등급 (Risk Level)
    def get_risk_level(p):
        if p <= 0.4: return 'Safe'
        elif p <= 0.7: return 'Attention'
        elif p <= 0.85: return 'Warning'
        else: return 'Critical'
        
    df['Risk Level'] = df['Probability'].apply(get_risk_level)
    
    # --- 3. 세분화 및 전략 태깅 (Priority Logic) ---
    
    # --- 3. 세분화 및 전략 태깅 (Enhanced Priority Logic) ---
    
    # 기준값 계산 (벡터 연산 위해 미리 계산)
    # 1. VIP 기준 (Bill Top 20%)
    bill_top_20 = df['total_bill'].quantile(0.8)
    bill_top_30 = df['total_bill'].quantile(0.7)
    
    # 2. Intl 기준 (Intl Charge Top 20%)
    intl_charge_top_20 = df['total_intl_charge'].quantile(0.8)
    
    # 3. Usage Drop 기준 (Day Minutes Bottom 50% - 완화됨)
    usage_bottom_50 = df['total_day_minutes'].quantile(0.5)
    
    def assign_strategy(row):
        # 전략 우선순위 (Priority)
        
        # 1. 🚨 VIP 전담 케어 (Highest Priority)
        # 조건: 이탈 확률 >= 85% AND 월 요금 상위 20%
        if (row['Probability'] >= 0.85) and (row['total_bill'] >= bill_top_20):
            return '🚨 VIP 전담 케어'
            
        # 2. 📞 불만 전담 마크 (CS Care)
        # 조건: CS 전화 >= 3회
        if row['number_customer_service_calls'] >= 3:
            return '📞 불만 전담 마크'
            
        # 3. 🌍 국제전화 요금제 제안 (Intl Upsell)
        # 조건: 국제전화 요금 상위 20% AND 플랜 없음
        is_intl_plan = (row['international_plan'] == 1) # 0/1 encoded
        if (row['total_intl_charge'] >= intl_charge_top_20) and is_intl_plan:
            return '🌍 국제전화 요금제 제안'
            
        # 4. 💰 요금 할인 쿠폰 발송 (Price Sensitive)
        # 조건: 월 요금 상위 30% AND Risk Level >= Warning (Warning, Critical)
        # Warning은 Probability > 0.70 -> 0.75로 상향 조정
        if (row['total_bill'] >= bill_top_30) and (row['Probability'] > 0.75):
            return '💰 요금 할인 쿠폰 발송'
            
        # 5. 일반 유지 관리 (General)
        return '일반 유지 관리'

    df['Strategy'] = df.apply(assign_strategy, axis=1)
    
    # --- 4. UI 구성 ---
    st.title("📊 경영진 및 마케팅 팀을 위한 이탈 관리 대시보드")
    
    # 사이드바: ROI 시뮬레이션 설정
    st.sidebar.markdown("### 🎛️ 시뮬레이션 설정")
    improvement_rate = st.sidebar.slider(
        "예상 이탈 개선율 (%)", 
        min_value=0, max_value=100, value=20, step=5
    )
    
    # A. KPI 보드
    st.markdown("### 1. 핵심 현황 (KPI)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    total_customers = len(df)
    # Warning(0.7 초과) 이상 고객 수 (Logic check: Attention=0.4~0.7, Warning=0.7~0.85, Critical=0.85+)
    warning_customers = len(df[df['Probability'] > 0.7]) 
    total_revenue_at_risk = df['risk_value'].sum()
    
    # ROI: 슬라이더 값 연동
    saved_revenue = total_revenue_at_risk * (improvement_rate / 100.0)
    
    kpi1.metric("총 관리 고객", f"{total_customers:,}명")
    kpi2.metric("집중 관리 대상 (Warning+)", f"{warning_customers:,}명", delta="요주의")
    kpi3.metric("총 기대 손실액", f"${total_revenue_at_risk:,.0f}")
    kpi4.metric(
        "캠페인 방어 효과 (ROI)", 
        f"${saved_revenue:,.0f}", 
        delta=f"이탈률 -{improvement_rate}% 가정"
    )
    
    st.markdown("---")
    
    # B. 현황 차트
    st.markdown("### 2. 고객 세분화 분석")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("위험 등급 분포")
        risk_counts = df['Risk Level'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values, 
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={
                'Safe': '#66bb6a', 'Attention': '#2196f3', 
                'Warning': '#ffa726', 'Critical': '#ff4b4b'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c2:
        st.subheader("주요 이탈 원인/전략 분포")
        # '일반 유지 관리'는 제외하고 시각화
        strategy_counts = df['Strategy'].value_counts().drop('일반 유지 관리', errors='ignore')
        fig_bar = px.bar(
            x=strategy_counts.index, 
            y=strategy_counts.values,
            color=strategy_counts.index,
            labels={'x': '전략 유형', 'y': '대상 고객 수'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")
    
    # C. 액션 테이블
    st.markdown("### 3. 실전 마케팅 리스트 (Actionable List)")
    
    # 필터링 옵션 가져오기 (전략 리스트 업데이트)
    all_strategies = df['Strategy'].unique()
    
    with st.expander("🔍 필터 옵션 열기", expanded=True):
        f1, f2 = st.columns(2)
        selected_risk = f1.multiselect(
            "위험 등급 선택", 
            ['Critical', 'Warning', 'Attention', 'Safe'],
            default=['Critical', 'Warning', 'Attention']
        )
        selected_strategy = f2.multiselect(
            "전략 유형 선택", 
            all_strategies,
            default=[x for x in all_strategies if 'VIP' in x or '불만' in x]
        )
    
    filtered_df = df.copy()
    if selected_risk:
        filtered_df = filtered_df[filtered_df['Risk Level'].isin(selected_risk)]
    if selected_strategy:
        filtered_df = filtered_df[filtered_df['Strategy'].isin(selected_strategy)]
        
    # 정렬 (기대 손실액 내림차순)
    filtered_df = filtered_df.sort_values(by='risk_value', ascending=False)
    
    # 표시할 컬럼 선택 및 정리
    display_df = filtered_df[['Risk Level', 'Probability', 'total_bill', 'risk_value', 'Strategy']].copy()
    display_df['Probability'] = display_df['Probability'] * 100 # 0-1 -> 0-100% 변환
    
    # 인덱스(고객ID 등)가 있다면 reset_index 하거나 그대로 사용
    
    st.dataframe(
        display_df,
        column_config={
            "Risk Level": "위험 등급",
            "Probability": st.column_config.ProgressColumn(
                "이탈 확률",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "total_bill": st.column_config.NumberColumn(
                "월 요금",
                format="$%.2f"
            ),
            "risk_value": st.column_config.NumberColumn(
                "기대 손실액 (Risk Value)",
                format="$%.2f"
            ),
            "Strategy": "추천 전략"
        },
        use_container_width=True,
        height=500
    )
    
    # CSV 다운로드 기능
    csv_data = display_df.to_csv(index=False).encode('utf-8-sig')
    
    st.download_button(
        label="📥 필터링된 리스트 다운로드 (CSV)",
        data=csv_data,
        file_name="churn_risk_list.csv",
        mime="text/csv"
    )

else:
    st.warning("데이터 또는 모델 로드에 실패했습니다.")
