import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob

# [필수] 페이지 설정은 무조건 맨 윗줄
st.set_page_config(page_title="유방암 분석기 (안전모드)", layout="wide", page_icon="🧬")

st.title("🧬 METABRIC 유방암 분석기")

# --- 1. 안전 장치 (머신러닝 라이브러리 체크) ---
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    ml_available = True
except ImportError:
    ml_available = False
    st.warning("⚠️ 'scikit-learn'이 설치되지 않아 AI 예측 기능이 꺼졌습니다. (requirements.txt 확인 필요)")

# --- 2. 데이터 로드 (Kaggle 제외, 업로드 파일 우선) ---
@st.cache_data
def load_data():
    # 1. 사용자가 올린 파일 찾기
    csv_files = glob.glob("*.csv")
    
    # METABRIC 파일 우선
    target = next((f for f in csv_files if "METABRIC" in f), None)
    
    # 없으면 아무 csv나 사용 (단, 인구 데이터 제외)
    if not target:
        target = next((f for f in csv_files if "202512" not in f), None)
        
    if target:
        return pd.read_csv(target, low_memory=False)
    return None

# --- 3. 사이드바 (입력창 강제 고정) ---
with st.sidebar:
    st.header("📂 데이터 & 입력")
    
    # 파일 업로더
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = load_data()

    if df is None:
        st.error("데이터가 없습니다. CSV 파일을 업로드해주세요.")
        st.stop()

    # 컬럼 매핑
    cols = df.columns.tolist()
    def find(k, c):
        for x in c:
            if k in x.lower(): return x
        return c[0]
        
    c_age = st.selectbox("나이 컬럼", cols, index=cols.index(find('age', cols)))
    c_size = st.selectbox("크기 컬럼", cols, index=cols.index(find('size', cols)))
    c_surv = st.selectbox("생존 컬럼", cols, index=cols.index(find('status', cols)))
    
    st.divider()
    
    # 입력창 (여기 있으면 무조건 보임)
    st.subheader("📝 환자 정보 입력")
    in_age = st.slider("나이 (Age)", 20, 100, 50)
    in_size = st.slider("종양 크기 (Size)", 0, 200, 20)
    
    run_btn = st.button("분석 실행", type="primary")

# --- 4. 메인 화면 로직 ---
# 전처리
df['Age'] = pd.to_numeric(df[c_age], errors='coerce')
df['Size'] = pd.to_numeric(df[c_size], errors='coerce')
df = df.dropna(subset=['Age', 'Size'])

# 생존 여부 처리 (ML용)
def parse_surv(x):
    s = str(x).lower()
    return 1 if 'liv' in s or '1' in s else 0
df['Target'] = df[c_surv].apply(parse_surv)

# 탭 구성
t1, t2 = st.tabs(["📊 시각화", "🤖 AI 예측"])

with t1:
    st.subheader("나의 위치 확인")
    c1, c2 = st.columns(2)
    
    # 나이 분포
    fig1 = px.histogram(df, x='Age', title="나이 분포")
    fig1.add_vline(x=in_age, line_color="red", annotation_text="나")
    c1.plotly_chart(fig1, use_container_width=True)
    
    # 크기 분포
    fig2 = px.histogram(df, x='Size', title="종양 크기 분포")
    fig2.add_vline(x=in_size, line_color="red", annotation_text="나")
    c2.plotly_chart(fig2, use_container_width=True)

with t2:
    if run_btn:
        if ml_available:
            if len(df) > 50:
                # 머신러닝 수행
                X = df[['Age', 'Size']]
                y = df['Target']
                
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X, y)
                prob = model.predict_proba([[in_age, in_size]])[0][1] * 100
                
                st.success(f"예측된 생존 확률: **{prob:.1f}%**")
                if prob < 50: st.error("위험군에 속할 가능성이 있습니다.")
                else: st.info("비교적 양호한 예후가 예상됩니다.")
            else:
                st.warning("데이터가 부족합니다.")
        else:
            st.error("라이브러리(scikit-learn) 문제로 AI 기능을 사용할 수 없습니다.")
    else:
        st.info("사이드바의 '분석 실행' 버튼을 눌러주세요.")
