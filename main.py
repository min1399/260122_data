import streamlit as st
import pandas as pd
import plotly.express as px
import kagglehub
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# [중요] 페이지 설정은 무조건 맨 처음에!
st.set_page_config(page_title="유방암 AI 분석기", layout="wide", page_icon="🧬")

st.title("🧬 METABRIC 유방암 AI 분석기 (안전 모드)")
st.caption("입력창을 사이드바로 옮겨 오류를 방지한 버전입니다.")

# --- 1. 데이터 로드 ---
@st.cache_data
def load_data():
    csv_files = glob.glob("*.csv")
    target_csvs = [f for f in csv_files if "METABRIC" in f]
    if target_csvs: return pd.read_csv(target_csvs[0], low_memory=False)
    
    try:
        path = kagglehub.dataset_download("gunesevitan/breast-cancer-metabric")
        files = glob.glob(os.path.join(path, "*.csv"))
        target = next((f for f in files if "METABRIC_RNA_Mutation" in f), files[0] if files else None)
        if target: return pd.read_csv(target, low_memory=False)
    except: pass
    return None

# --- 2. 사이드바 설정 (여기에 입력창 배치) ---
with st.sidebar:
    st.header("1. 데이터 파일")
    uploaded_file = st.file_uploader("CSV 업로드", type=['csv'])
    df = pd.read_csv(uploaded_file) if uploaded_file else load_data()
    
    if df is None:
        st.error("데이터 없음")
        st.stop()

    st.header("2. 컬럼 매핑")
    cols = df.columns.tolist()
    def find(k, c):
        for x in c: 
            if k in x.lower(): return x
        return c[0]
        
    c_age = st.selectbox("나이", cols, index=cols.index(find('age', cols)))
    c_size = st.selectbox("크기", cols, index=cols.index(find('size', cols)))
    c_surv = st.selectbox("생존여부", cols, index=cols.index(find('status', cols)))
    
    st.divider()
    
    # [핵심] 환자 정보 입력을 사이드바로 이동 (오류가 나도 보임)
    st.header("3. 환자 정보 입력 (AI 예측용)")
    input_age = st.slider("환자 나이 (Age)", 20, 100, 50)
    input_size = st.slider("종양 크기 (Size)", 0, 200, 20)
    
    run_predict = st.button("AI 생존율 예측하기", type="primary")

# --- 3. 데이터 전처리 ---
df_clean = df.copy()
df_clean['Age'] = pd.to_numeric(df_clean[c_age], errors='coerce')
df_clean['Size'] = pd.to_numeric(df_clean[c_size], errors='coerce')

# 생존 여부 (Living/Deceased -> 1/0)
def parse_status(x):
    s = str(x).lower()
    if 'liv' in s or '1' in s: return 1
    return 0
df_clean['Target'] = df_clean[c_surv].apply(parse_status)
df_clean = df_clean.dropna(subset=['Age', 'Size', 'Target'])

# --- 4. 메인 화면 ---

# (1) 데이터 통계
c1, c2, c3 = st.columns(3)
c1.metric("데이터 수", f"{len(df_clean):,}명")
c2.metric("평균 나이", f"{df_clean['Age'].mean():.1f}세")
c3.metric("생존율", f"{df_clean['Target'].mean()*100:.1f}%")

st.divider()

# (2) 시각화 (나의 위치)
st.subheader("📊 나의 위치 확인")
c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(df_clean, x='Age', title="나이 분포")
    fig.add_vline(x=input_age, line_color="red", line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.histogram(df_clean, x='Size', title="크기 분포")
    fig.add_vline(x=input_size, line_color="red", line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

# (3) AI 예측 결과 (버튼 누르면 실행)
if run_predict:
    st.divider()
    st.subheader("🤖 AI 생존 예측 결과")
    
    if len(df_clean) > 50:
        # 모델 학습
        X = df_clean[['Age', 'Size']]
        y = df_clean['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 예측
        prob = model.predict_proba([[input_age, input_size]])[0][1] * 100
        
        st.write(f"학습 정확도: {accuracy_score(y_test, model.predict(X_test))*100:.1f}%")
        
        # 결과 카드
        if prob >= 70:
            color = "green"
            msg = "긍정적 (Good)"
        elif prob >= 40:
            color = "orange"
            msg = "보통 (Moderate)"
        else:
            color = "red"
            msg = "주의 (Risk)"
            
        st.markdown(f"""
        <div style="padding:20px; border:2px solid {color}; border-radius:10px; text-align:center;">
            <h3>예상 생존 확률</h3>
            <h1 style="color:{color};">{prob:.1f}%</h1>
            <p>{msg}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("데이터가 부족하여 예측할 수 없습니다.")
