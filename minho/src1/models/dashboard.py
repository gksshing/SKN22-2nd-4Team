import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform


# -----------------------------------------------------------------------------
# 0. 한글 폰트 및 설정
# -----------------------------------------------------------------------------
def set_korean_font():
    system_name = platform.system()
    if system_name == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif system_name == 'Darwin': # Mac
        plt.rc('font', family='AppleGothic')
    else:
        plt.rc('font', family='sans-serif')
    plt.rc('axes', unicode_minus=False)

set_korean_font()

# -----------------------------------------------------------------------------
# 1. 페이지 설정 및 스타일링
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Diagnosis Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 10px;
    }
    .big-number { font-size: 2.2rem; font-weight: bold; color: #2c3e50; }
    .loss-number { font-size: 2.2rem; font-weight: bold; color: #e74c3c; }
    .risk-row {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .risk-title { font-weight: bold; font-size: 1.1rem; color: #2c3e50; }
    .risk-stat { font-weight: bold; font-size: 1.2rem; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # 데이터 경로 확인 (기본 경로 설정)
    DATA_PATH = "data/01_raw/train.csv"
    
    # 파일이 없으면 현재 디렉토리에서 찾기 (유연성 확보)
    if not os.path.exists(DATA_PATH):
        if os.path.exists("train.csv"):
            DATA_PATH = "train.csv"
        else:
            return None

    df = pd.read_csv(DATA_PATH)

    # 전처리
    if 'international_plan' in df.columns:
        df['international_plan'] = (df['international_plan'] == 'yes').astype(int)
    if 'voice_mail_plan' in df.columns:
        df['voice_mail_plan'] = (df['voice_mail_plan'] == 'yes').astype(int)
    
    # Target 변환
    if 'churn' in df.columns and df['churn'].dtype == object:
        df['churn'] = df['churn'].apply(lambda x: 1 if x == 'yes' else 0)

    # 파생 변수: 총 매출 (Revenue) 추정
    charge_cols = ['total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge']
    df['total_revenue'] = df[charge_cols].sum(axis=1)
    
    return df

df = load_data()

# 데이터 로드 실패 시 중단
if df is None:
    st.error("데이터 파일(train.csv)을 찾을 수 없습니다. 파일을 업로드해주세요.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. 사이드바 네비게이션
# -----------------------------------------------------------------------------
st.sidebar.title("🛡️ Churn Diagnosis")
st.sidebar.info("고객이탈 진단 및 방지 솔루션")
page = st.sidebar.radio("MENU", ["1. 현황 진단", "2. 솔루션 & 시뮬레이션", "3. 기대 효과"])

st.sidebar.markdown("---")
@st.cache_data
def get_exchange_rate(pair="KRW=X"):
    try:
        ticker = yf.Ticker(pair)
        # 최근 1일치 데이터 중 종가(Close)를 가져옴
        rate = ticker.history(period="1d")['Close'].iloc[-1]
        return rate
    except Exception as e:
        st.error(f"환율 정보를 가져오는데 실패했습니다: {e}")
        return 1200.0  # 에러 발생 시 기본값 설정 (예: 1200원)

# --- 사이드바 UI 변경 ---
st.sidebar.header("설정")

# 기존 text_input 대신 라디오 버튼으로 화폐 선택 기능 구현
currency_option = st.sidebar.radio(
    "화폐 단위 선택",
    ("USD ($)", "KRW (₩) - 실시간 환율 적용")
)

# --- 환율 적용 로직 ---
if "KRW" in currency_option:
    current_rate = get_exchange_rate("KRW=X") # USD -> KRW 환율 조회
    currency_symbol = "₩"
    exchange_factor = current_rate
    st.sidebar.success(f"현재 적용 환율: 1 USD = {current_rate:,.2f} KRW")
else:
    currency_symbol = int("$")
    exchange_factor = 1.0

# -----------------------------------------------------------------------------
# 4. 페이지별 로직
# -----------------------------------------------------------------------------

# === Page 1: 현황 진단 ===
if page == "1. 현황 진단":
    st.title("🩺 고객 이탈 현황 및 핵심 원인 진단")
    st.markdown("현재 회사의 데이터 분석 결과, **3가지 주요 원인**이 이탈을 주도하고 있습니다.")

    # KPI Calculation
    total_customers = len(df)
    churn_count = df['churn'].sum()
    churn_rate = churn_count / total_customers * 100
    total_revenue = df['total_revenue'].sum()
    lost_revenue = df[df['churn'] == 1]['total_revenue'].sum()

    # Top KPI Display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h4>총 고객 수</h4><div class='big-number'>{total_customers:,.0f}명</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h4>이탈률 (Churn Rate)</h4><div class='loss-number'>{churn_rate:.1f}%</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h4>월 총 매출</h4><div class='big-number'>{currency_symbol}{total_revenue:,.0f}</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><h4>이탈 손실액</h4><div class='loss-number'>{currency_symbol}{lost_revenue:,.0f}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🚨 3대 핵심 이탈 원인 분석")
    st.markdown("각 원인별 고위험군을 정의하고, 해당 그룹의 **이탈률**과 **매출 손실 기여도**를 산출했습니다.")

    # Header Row
    st.markdown("""
    <div style="display: flex; justify-content: space-between; padding: 10px; border-bottom: 2px solid #ddd; font-weight: bold; color: #555;">
        <div style="width: 40%;">📌 리스크 요인 (Risk Factor)</div>
        <div style="width: 30%; text-align: center;">📉 그룹 이탈률 (vs 평균)</div>
        <div style="width: 30%; text-align: right;">💸 손실 기여액</div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Risk Factor 1: 고객센터 전화 (CS Calls >= 4)
    # -------------------------------------------------------------------------
    cs_risk_group = df[df['number_customer_service_calls'] >= 4]
    cs_churn_rate = cs_risk_group['churn'].mean() * 100 if len(cs_risk_group) > 0 else 0
    cs_loss = cs_risk_group[cs_risk_group['churn'] == 1]['total_revenue'].sum()

    st.markdown(f"""
    <div class='risk-row' style="display: flex; align-items: center; justify-content: space-between;">
        <div style="width: 40%;">
            <div class='risk-title'>① 고객센터 전화 연결 과다</div>
            <div style="font-size: 0.9em; color: gray;">기준: 고객센터 통화 4회 이상</div>
        </div>
        <div style="width: 30%; text-align: center;">
            <div class='risk-stat' style="color: #e74c3c;">{cs_churn_rate:.1f}%</div>
            <div style="font-size: 0.8em; color: gray;">(평균 대비 {cs_churn_rate/churn_rate:.1f}배)</div>
        </div>
        <div style="width: 30%; text-align: right;">
            <div class='risk-stat'>{currency_symbol}{cs_loss:,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Risk Factor 2: 국제전화 (International Plan == 1)
    # -------------------------------------------------------------------------
    intl_risk_group = df[df['international_plan'] == 1]
    intl_churn_rate = intl_risk_group['churn'].mean() * 100 if len(intl_risk_group) > 0 else 0
    intl_loss = intl_risk_group[intl_risk_group['churn'] == 1]['total_revenue'].sum()

    st.markdown(f"""
    <div class='risk-row' style="display: flex; align-items: center; justify-content: space-between;">
        <div style="width: 40%;">
            <div class='risk-title'>② 국제전화 요금제 가입자</div>
            <div style="font-size: 0.9em; color: gray;">기준: International Plan 가입 고객</div>
        </div>
        <div style="width: 30%; text-align: center;">
            <div class='risk-stat' style="color: #e74c3c;">{intl_churn_rate:.1f}%</div>
            <div style="font-size: 0.8em; color: gray;">(평균 대비 {intl_churn_rate/churn_rate:.1f}배)</div>
        </div>
        <div style="width: 30%; text-align: right;">
            <div class='risk-stat'>{currency_symbol}{intl_loss:,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Risk Factor 3: 주간 통화량 (Day Minutes > 260) -> NEW
    # -------------------------------------------------------------------------
    # 통상적으로 260분 이상 사용 시 요금 부담으로 이탈률 급증 (데이터셋 특성 반영)
    day_risk_group = df[df['total_day_minutes'] > 260]
    day_churn_rate = day_risk_group['churn'].mean() * 100 if len(day_risk_group) > 0 else 0
    day_loss = day_risk_group[day_risk_group['churn'] == 1]['total_revenue'].sum()

    st.markdown(f"""
    <div class='risk-row' style="display: flex; align-items: center; justify-content: space-between;">
        <div style="width: 40%;">
            <div class='risk-title'>③ 주간 통화량 과다 (헤비 유저)</div>
            <div style="font-size: 0.9em; color: gray;">기준: 주간 통화 260분 이상</div>
        </div>
        <div style="width: 30%; text-align: center;">
            <div class='risk-stat' style="color: #e74c3c;">{day_churn_rate:.1f}%</div>
            <div style="font-size: 0.8em; color: gray;">(평균 대비 {day_churn_rate/churn_rate:.1f}배)</div>
        </div>
        <div style="width: 30%; text-align: right;">
            <div class='risk-stat'>{currency_symbol}{day_loss:,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# === Page 2: 솔루션 & 시뮬레이션 ===
elif page == "2. 솔루션 & 시뮬레이션":
    st.title("💊 3단계 이탈 방지 시뮬레이터")
    st.markdown("각 원인별 맞춤형 전략을 수행했을 때, 회복 가능한 매출을 예측합니다.")

    col_simulator, col_results = st.columns([1, 1])

    # --- 1. 왼쪽: 시뮬레이터 조작 ---
    with col_simulator:
        st.markdown("### 🎛️ 전략별 방어율 설정")
        st.info("각 전략 실행 시 예상되는 **이탈 방어 성공률(%)**을 조정하세요.")
        
        # Slider 1: CS
        st.markdown("**1️⃣ CS 불만 케어 프로그램**")
        improvement_cs = st.slider("CS 케어 성공률 (%)", 0, 100, 30, step=5, key="slide1")
        
        # Slider 2: International
        st.markdown("**2️⃣ 국제전화 전용 할인 오퍼**")
        improvement_intl = st.slider("국제전화 케어 성공률 (%)", 0, 100, 20, step=5, key="slide2")
        
        # Slider 3: Day Minutes (New)
        st.markdown("**3️⃣ 헤비 유저(주간 통화) 요금제 컨설팅**")
        improvement_day = st.slider("주간 통화 케어 성공률 (%)", 0, 100, 25, step=5, key="slide3")

    # --- 시뮬레이션 로직 ---
    df_sim = df.copy()
    
    # Logic 1: CS Calls >= 3 (엄격한 기준 적용)
    idx_cs = df_sim[(df_sim['number_customer_service_calls'] >= 3) & (df_sim['churn'] == 1)].index
    if len(idx_cs) > 0:
        save_count = int(len(idx_cs) * (improvement_cs / 100))
        saved_idx = np.random.choice(idx_cs, save_count, replace=False)
        df_sim.loc[saved_idx, 'churn'] = 0
        
    # Logic 2: International Plan
    idx_intl = df_sim[(df_sim['international_plan'] == 1) & (df_sim['churn'] == 1)].index
    if len(idx_intl) > 0:
        save_count = int(len(idx_intl) * (improvement_intl / 100))
        # 이미 0으로 바뀐 사람은 제외하지 않고 덮어씌움 (독립적 캠페인 가정)
        saved_idx = np.random.choice(idx_intl, save_count, replace=False)
        df_sim.loc[saved_idx, 'churn'] = 0

    # Logic 3: Day Minutes > 260
    idx_day = df_sim[(df_sim['total_day_minutes'] > 260) & (df_sim['churn'] == 1)].index
    if len(idx_day) > 0:
        save_count = int(len(idx_day) * (improvement_day / 100))
        saved_idx = np.random.choice(idx_day, save_count, replace=False)
        df_sim.loc[saved_idx, 'churn'] = 0

    # 결과 계산
    new_lost_revenue = df_sim[df_sim['churn'] == 1]['total_revenue'].sum()
    original_lost_revenue = df[df['churn'] == 1]['total_revenue'].sum()
    recovered_revenue = original_lost_revenue - new_lost_revenue
    
    new_churn_rate = df_sim['churn'].mean() * 100
    original_churn_rate = df['churn'].mean() * 100

    # --- 2. 오른쪽: 결과 시각화 ---
    with col_results:
        st.markdown("### 🚀 시뮬레이션 결과")
        
        # Metrics
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="📉 예상 이탈률", 
                      value=f"{new_churn_rate:.2f}%", 
                      delta=f"{new_churn_rate - original_churn_rate:.2f}%p",
                      delta_color="inverse")
        with m_col2:
            st.metric(label="💰 월 매출 회복", 
                      value=f"{currency_symbol}{recovered_revenue:,.0f}", 
                      delta=f"{(recovered_revenue/original_lost_revenue)*100:.1f}% 회복",
                      delta_color="normal")
        
        st.write("")
        
        # Matplotlib Graph
        fig, ax = plt.subplots(figsize=(6, 4))
        x_labels = ['Before (현재)', 'After (개선후)']
        y_values = [original_churn_rate, new_churn_rate]
        colors = ['#95a5a6', '#2ecc71'] 
        
        bars = ax.bar(x_labels, y_values, color=colors, width=0.5)
        ax.set_ylabel('이탈률 (%)')
        ax.set_ylim(0, max(y_values)*1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
        st.pyplot(fig)

    # --- 3. 하단: Action Plan (3 Columns) ---
    st.markdown("---")
    st.subheader("💡 3대 영역별 세부 실행 계획 (Action Plan)")
    
    col_act1, col_act2, col_act3 = st.columns(3)
    
    with col_act1:
        st.error("📞 ① 고객센터 (CS)")
        st.markdown("""
        - **우선 상담:** 3회 이상 연결 시도 시 VIP 라인 자동 배정
        - **해피콜:** 불만 접수 24시간 내 매니저 직접 통화
        """)
        
    with col_act2:
        st.warning("✈️ ② 국제전화 (Intl)")
        st.markdown("""
        - **전용 요금:** 국제전화 50% 할인 부가서비스 무료 체험
        - **가족 결합:** 해외 체류 가족 등록 시 무료 통화 제공
        """)

    with col_act3:
        st.info("☀️ ③ 주간 통화 (Day)")
        st.markdown("""
        - **헤비 유저 요금제:** 무제한 요금제 업셀링 (약정 할인)
        - **타겟 쿠폰:** 주간 사용량이 피크일 때 데이터 쿠폰 발송
        """)

# === Page 3: 기대 효과 ===
elif page == "3. 기대 효과":
    st.title("📈 To-Be: 전략 도입 후 미래 예측")
    st.markdown("3가지 솔루션이 안착되었을 때 기대되는 회사의 연간 재무적 변화입니다.")

    # KPI 설정 (가정치)
    current_churn = 14.1
    target_churn = 9.5 # 3가지 전략 성공 시 더 낮아짐 가정
    
    # 간단한 연산
    loss_per_month = 39000 # 대략적 수치
    projected_loss = loss_per_month * (target_churn / current_churn)
    annual_save = (loss_per_month - projected_loss) * 12

    col_final1, col_final2 = st.columns(2)
    
    with col_final1:
        st.markdown("### 📊 연간 매출 증대 효과")
        st.markdown(f"""
        <div style='font-size: 3rem; color: #27ae60; font-weight: bold;'>
        +{currency_symbol}{annual_save:,.0f}
        </div>
        <div style='color: gray;'>Yearly Revenue Recovered</div>
        """, unsafe_allow_html=True)
        
    with col_final2:
        st.markdown("### 📉 목표 이탈률 달성")
        st.markdown(f"""
        <div style='font-size: 3rem; color: #2980b9; font-weight: bold;'>
        {target_churn}%
        </div>
        <div style='color: gray;'>Target Churn Rate (from {current_churn}%)</div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### ✅ 경영진을 위한 제언 (Executive Summary)
    1. **고객센터 불만**은 단순 응대가 아닌 **'시스템적 방어(우선 배정)'**가 필요합니다.
    2. **국제전화 사용자**는 경쟁사로 넘어가기 가장 쉬운 그룹이므로 **'가격 혜택'**이 필수입니다.
    3. **주간 통화량이 많은 헤비 유저**는 우리 회사의 VIP이므로, **'요금제 컨설팅'**을 통해 락인(Lock-in) 해야 합니다.
    """)