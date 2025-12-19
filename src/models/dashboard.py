import streamlit as st
import pandas as pd
import seaborn as sns
import pickle
import os
from catboost import CatBoostClassifier
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

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
    
    # 1. 전처리 (글로벌 전처리) - df에 바로 적용하여 모델 입력 및 전략 로직 모두 해결
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
    
    # --- 3. 세분화 및 전략 태깅 (우선순위 로직) ---
    
    # --- 3. 세분화 및 전략 태깅 (강화된 우선순위 로직) ---
    
    # 기준값 계산 (벡터 연산 위해 미리 계산)
    # 1. VIP 기준 (Bill Top 20%)
    bill_top_20 = df['total_bill'].quantile(0.8)
    bill_top_30 = df['total_bill'].quantile(0.7)
    
    # 2. Intl 기준 (Intl Charge Top 20%)
    intl_charge_top_20 = df['total_intl_charge'].quantile(0.8)
    
    # 3. 사용량 감소 기준 (Day Minutes 하위 50% - 완화됨)
    usage_bottom_50 = df['total_day_minutes'].quantile(0.5)
    
    def assign_strategy(row):
        # 전략 우선순위 (Priority)
        
        # 1. 🚨 VIP 전담 케어 (Highest Priority)
        # 조건: 이탈 확률 >= 85% AND 월 요금 상위 20%
        if (row['Probability'] >= 0.85) and (row['total_bill'] >= bill_top_20):
            return '🚨 VIP 전담 케어'
            
        # 2. 📞 불만 전담 마크 (CS 케어)
        # 조건: CS 전화 >= 3회
        if row['number_customer_service_calls'] >= 3:
            return '📞 불만 전담 마크'
            
        # 3. 🌍 국제전화 요금제 제안 (국제전화 업셀링)
        # 조건: 국제전화 요금 상위 20% AND 플랜 없음
        is_intl_plan = (row['international_plan'] == 1) # 0/1 인코딩됨
        if (row['total_intl_charge'] >= intl_charge_top_20) and is_intl_plan:
            return '🌍 국제전화 요금제 제안'
            
        # 4. 💰 요금 할인 쿠폰 발송 (가격 민감형)
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
    kpi3.metric("총 기대 손실액", f"€{total_revenue_at_risk:,.0f}")
    kpi4.metric(
        "캠페인 방어 효과 (ROI)", 
        f"€{saved_revenue:,.0f}", 
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
        
        # 탭 구성: 고객 수 vs 금액(Risk Value)
        tab1, tab2 = st.tabs(["👥 대상 고객 수", "💰 전략별 기대 손실액"])
        
        # 공통 필터: '일반 유지 관리' 제외
        chart_df = df[df['Strategy'] != '일반 유지 관리']
        
        # --- Tab 1: 기존 카운트 차트 ---
        with tab1:
            strategy_counts = chart_df['Strategy'].value_counts()
            fig_bar = px.bar(
                x=strategy_counts.index, 
                y=strategy_counts.values,
                color=strategy_counts.index,
                labels={'x': '전략 유형', 'y': '대상 고객 수'}
            )
            # 탭 내부 차트 높이 등 조정 가능
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # --- Tab 2: 기대 손실액 (Stacked by Risk Level) ---
        with tab2:
            # 전략 및 위험 등급별로 그룹화하여 합계 계산
            risk_agg = chart_df.groupby(['Strategy', 'Risk Level'], as_index=False)['risk_value'].sum()
            
            # Risk Level 순서 정렬 (Critical이 위로 오거나 강조되도록)
            risk_order = ['Safe', 'Attention', 'Warning', 'Critical']
            
            fig_revenue = px.bar(
                risk_agg,
                x='Strategy',
                y='risk_value',
                color='Risk Level',
                category_orders={'Risk Level': risk_order}, # 범례 순서 고정
                color_discrete_map={
                    'Safe': '#66bb6a', 'Attention': '#2196f3', 
                    'Warning': '#ffa726', 'Critical': '#ff4b4b'
                },
                labels={'risk_value': '기대 손실액 (€)', 'Strategy': '전략 유형'}
            )
            
            # 포맷팅 및 디자인 개선
            fig_revenue.update_traces(hovertemplate='%{y:€,.0f}') # 툴팁: €표시 및 천단위 콤마, 소수점 제거
            
            # 총합 텍스트 추가를 위한 데이터 계산
            total_rev = chart_df.groupby('Strategy', as_index=False)['risk_value'].sum()
            
            # Scatter Trace로 텍스트 추가 (누적 막대 차트 위에 표시)
            fig_revenue.add_trace(
                go.Scatter(
                    x=total_rev['Strategy'], 
                    y=total_rev['risk_value'],
                    text=total_rev['risk_value'],
                    mode='text',
                    texttemplate='%{text:€,.0f}', # 텍스트 포맷: €1,234
                    textposition='top center',
                    showlegend=False,
                    hoverinfo='skip'
                )
            )

            fig_revenue.update_layout(
                yaxis_tickformat='€2s',   # 축 단위 표시 (K, M)
                xaxis={'categoryorder':'total descending'}, # 1. 막대 정렬: 총합 기준 내림차순
                bargap=0.4, # 2. 디자인: 막대 두께를 얇게
                margin=dict(t=50) # 상단 여백 확보 (텍스트 잘림 방지)
            )
            
            st.plotly_chart(fig_revenue, use_container_width=True)
        
    st.markdown("---")
    
    # 전략 가이드 섹션
    with st.expander("ℹ️ 용어 가이드: 위험 등급과 마케팅 전략 용어 풀이", expanded=False):
        st.markdown("""
        ### 1. 🚦 위험 등급 정의 (Risk Levels)
        - **🔴 Critical (위험):** 이탈 확률 **85% 초과**. 즉각적인 조치가 필요한 최고 위험군.
        - **🟠 Warning (주의):** 이탈 확률 **70% ~ 85%**. 이탈 징후가 뚜렷하여 집중 관리 필요.
        - **🟡 Attention (관심):** 이탈 확률 **40% ~ 70%**. 케어가 필요한 잠재적 위험군.
        - **🟢 Safe (안정):** 이탈 확률 **40% 이하**. 안정적인 장기 충성 고객.

        ---

        ### 2. 🏹 마케팅 전략 가이드 (Marketing Strategies)
        - **🚨 VIP 전담 케어:**
            - **대상:** 이탈 확률 85% 이상(Critical) + 월 요금 상위 20% (높은 요금)
            - **설명:** 놓치면 매출 타격이 큰 최상위 핵심 고객입니다. 무조건 잡아야 합니다.

        - **📞 불만 전담 마크:**
            - **대상:** 고객센터 전화(CS Calls) 3회 이상
            - **설명:** 서비스에 대한 불만이 누적된 상태입니다. 선제적 해피콜로 불만을 해소해야 합니다.

        - **🌍 국제전화 요금제 제안:**
            - **대상:** 국제전화 요금 상위 20% + 국제전화 전용 플랜 미가입
            - **설명:** 국제전화를 비싸게 쓰고 있는 고객입니다. 할인 요금제로 유도하면(업셀링) 만족도가 올라갑니다.

        - **💰 요금 할인 쿠폰 발송:**
            - **대상:** 월 요금 상위 30% + 이탈 위험도 'Warning' 이상
            - **설명:** 특별한 불만은 없으나 요금 부담이나 타사 프로모션 때문에 흔들리는 고객입니다. 가격 혜택이 필요합니다.
        """)
        
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
                format="€%.2f"
            ),
            "risk_value": st.column_config.NumberColumn(
                "기대 손실액 (Risk Value)",
                format="€%.2f"
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
