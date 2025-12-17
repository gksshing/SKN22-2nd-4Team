import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import pickle
from catboost import CatBoostClassifier

# --- 1. 페이지 설정 및 모델 로드 ---
st.set_page_config(
    page_title="고객 이탈 예측 대시보드",
    page_icon="📊",
    layout="wide"
)

st.title("📊 고객 이탈 예측 시스템 (Churn Prediction)")
st.markdown("---")

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "churn_model.cbm")
FEATURES_PATH = os.path.join(CURRENT_DIR, "features.pkl")

# 모델 및 데이터 로드 (캐싱 사용)
@st.cache_resource
def load_model_and_features():
    # 1. 모델 로드
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    
    # 2. Feature Names 로드
    with open(FEATURES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
        
    return model, feature_names

# 로딩 중 표시
with st.spinner("모델 및 데이터를 로딩 중입니다..."):
    # 파일 존재 여부 확인
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        st.error("모델 파일 또는 피처 파일이 없습니다. 'save_model.py'를 먼저 실행해주세요.")
        st.stop()
        
    model, feature_names = load_model_and_features()

# --- 2. 사이드바: 사용자 입력 (User Input) ---
st.sidebar.header("📝 고객 정보 입력")

# 입력값을 저장할 딕셔너리
user_input = {}

# 그룹 1: 기본 가입 정보 (Demographics & Plans)
with st.sidebar.expander("👤 기본 가입 정보", expanded=True):
    # State 선택
    state_options = ['KS', 'OH', 'NJ', 'OK', 'AL', 'MA', 'MO', 'LA', 'WV', 'IN'] # 예시
    user_input['state'] = st.selectbox("State (주)", state_options)
    
    user_input['account_length'] = st.number_input("가입 기간 (일)", min_value=1, value=100)
    user_input['area_code'] = st.selectbox("지역 코드 (Area Code)", ["area_code_408", "area_code_415", "area_code_510"])
    
    # Yes/No 입력 -> 1/0 변환
    intl_plan = st.radio("국제전화 플랜 가입", ["Yes", "No"])
    user_input['international_plan'] = 1 if intl_plan == "Yes" else 0
    
    vmail_plan = st.radio("음성메일 플랜 가입", ["Yes", "No"])
    user_input['voice_mail_plan'] = 1 if vmail_plan == "Yes" else 0
    
    user_input['number_vmail_messages'] = st.number_input("음성메일 메시지 수", min_value=0, value=0)

# 그룹 2: 통화량 정보 (Call Usage)
with st.sidebar.expander("📞 통화 사용량 정보", expanded=False):
    st.markdown("**주간 (Day)**")
    user_input['total_day_minutes'] = st.number_input("주간 통화 분(Min)", min_value=0.0, value=150.0)
    user_input['total_day_calls'] = st.number_input("주간 통화 횟수", min_value=0, value=100)
    user_input['total_day_charge'] = st.number_input("주간 요금", min_value=0.0, value=25.0)
    
    st.markdown("**저녁 (Evening)**")
    user_input['total_eve_minutes'] = st.number_input("저녁 통화 분(Min)", min_value=0.0, value=200.0)
    user_input['total_eve_calls'] = st.number_input("저녁 통화 횟수", min_value=0, value=100)
    user_input['total_eve_charge'] = st.number_input("저녁 요금", min_value=0.0, value=17.0)
    
    st.markdown("**야간 (Night)**")
    user_input['total_night_minutes'] = st.number_input("야간 통화 분(Min)", min_value=0.0, value=200.0)
    user_input['total_night_calls'] = st.number_input("야간 통화 횟수", min_value=0, value=100)
    user_input['total_night_charge'] = st.number_input("야간 요금", min_value=0.0, value=9.0)
    
    st.markdown("**국제 (Intl)**")
    user_input['total_intl_minutes'] = st.number_input("국제 통화 분(Min)", min_value=0.0, value=10.0)
    user_input['total_intl_calls'] = st.number_input("국제 통화 횟수", min_value=0, value=3)
    user_input['total_intl_charge'] = st.number_input("국제 요금", min_value=0.0, value=2.7)

# 그룹 3: 기타 고객 서비스
with st.sidebar.expander("🎧 고객 서비스 (CS)", expanded=False):
    user_input['number_customer_service_calls'] = st.number_input("고객센터 전화 횟수", min_value=0, max_value=20, value=1)


# 입력 데이터를 DataFrame으로 변환
input_df = pd.DataFrame([user_input])

# 중요: 학습된 모델의 Feature 순서와 동일하게 정렬
# 없는 컬럼은 0으로 채우고, 불필요한 컬럼은 제거
# (현재 예시 UI에서는 모든 피처를 다 받지 않았을 수 있으므로 안전장치 추가)
for col in feature_names:
    if col not in input_df.columns:
        # UI에서 입력받지 않은 값이 있다면 기본값 0 처리 (혹은 적절한 값)
        input_df[col] = 0

# 최종적으로 Feature Names 순서대로 정렬
input_df = input_df[feature_names]


# --- 3. 메인 화면: 예측 결과 ---

# 예측 수행
# predict_proba 반환값은 [class0_prob, class1_prob]
prob_churn = model.predict_proba(input_df)[0][1] # 이탈(1) 확률
prob_percent = prob_churn * 100

# 화면 레이아웃 분할 (왼쪽: 게이지, 오른쪽: 상세 정보)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("이탈 위험도 (Churn Risk)")
    
    # 게이지 차트 생성
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob_percent,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "이탈 확률(%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps' : [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold' : {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prob_percent
            }
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # 상태 메시지 표시
    if prob_percent < 30:
        st.success("✅ 안전 (Safe): 이탈 가능성이 낮습니다.")
    elif prob_percent < 70:
        st.warning("⚠️ 주의 (Warning): 관리가 필요합니다.")
    else:
        st.error("🚨 위험 (Danger): 적극적인 개입이 시급합니다!")


with col2:
    st.subheader("주요 이탈 요인 (Feature Importance)")
    
    # Feature Importance 추출
    importances = model.get_feature_importance()
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)
    
    # 막대 그래프
    fig_bar = px.bar(
        feature_imp, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        text='Importance',
        color='Importance',
        color_continuous_scale='Reds'
    )
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# --- 4. What-If 시뮬레이터 ---
st.header("🤔 What-If 시뮬레이터")
st.markdown("특정 변수를 변화시켰을 때 이탈 확률이 어떻게 변하는지 확인해보세요.")

sim_col1, sim_col2 = st.columns(2)

with sim_col1:
    st.markdown("#### 💬 고객센터 전화 횟수 변경")
    # 현재 값
    current_calls = user_input['number_customer_service_calls']
    
    # 슬라이더로 변경해보기
    new_calls = st.slider("전화 횟수를 변경해보세요:", min_value=0, max_value=20, value=current_calls)
    
    # 예측해보기
    sim_input = input_df.copy()
    sim_input['number_customer_service_calls'] = new_calls
    
    sim_prob = model.predict_proba(sim_input)[0][1] * 100
    delta = sim_prob - prob_percent
    
    st.metric(
        label="예상 이탈 확률", 
        value=f"{sim_prob:.2f}%", 
        delta=f"{delta:.2f}%p",
        delta_color="inverse"
    )

with sim_col2:
    st.markdown("#### ✈️ 국제전화 플랜 변경")
    # 현재 상태 반전
    current_plan = user_input['international_plan']
    new_plan = 1 - current_plan # 0이면 1, 1이면 0
    
    btn_label = "플랜 가입하기" if current_plan == 0 else "플랜 해지하기"
    
    if st.button(btn_label):
        sim_input_plan = input_df.copy()
        sim_input_plan['international_plan'] = new_plan
        
        sim_prob_plan = model.predict_proba(sim_input_plan)[0][1] * 100
        delta_plan = sim_prob_plan - prob_percent
        
        st.metric(
            label="변경 후 이탈 확률",
            value=f"{sim_prob_plan:.2f}%",
            delta=f"{delta_plan:.2f}%p",
            delta_color="inverse"
        )
    else:
        st.info("버튼을 눌러 시뮬레이션을 실행해보세요.")
