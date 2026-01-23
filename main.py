import streamlit as st
import pandas as pd
import plotly.express as px
import kagglehub
import os
import glob
# 머신러닝 라이브러리 추가
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. 페이지 설정 및 스타일링 ---
st.set_page_config(
    page_title="METABRIC 유방암 AI 분석기",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stCode { font-family: 'D2Coding', 'Courier New', monospace; }
    .prediction-card {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #4b92ff;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 METABRIC Breast Cancer AI Analysis")
st.caption("유방암 임상 데이터 시각화 및 머신러닝 생존 예측")
st.markdown("---")

# 탭 구성 (AI 예측 탭 추가됨)
tab1, tab2, tab3, tab4 = st.tabs(["📊 대시보드", "🤖 AI 생존 예측", "💻 코드 분석", "📚 데이터 가이드"])

# --- 데이터 로드 함수 ---
@st.cache_data
def load_data():
    csv_files = glob.glob("*.csv")
    # 1. METABRIC 파일 우선 탐색
    target_csvs = [f for f in csv_files if "METABRIC" in f]
    if not target_csvs:
        target_csvs = [f for f in csv_files if "202512" not in f]
    
    if target_csvs:
        return pd.read_csv(target_csvs[0], low_memory=False)
    
    try:
        path = kagglehub.dataset_download("gunesevitan/breast-cancer-metabric")
        files = glob.glob(os.path.join(path, "*.csv"))
        target = next((f for f in files if "METABRIC_RNA_Mutation" in f), files[0] if files else None)
        if target:
            return pd.read_csv(target, low_memory=False)
    except Exception:
        pass
    return None

# 사이드바 데이터 로드
with st.sidebar:
    st.header("📂 Data Controller")
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=['csv'])
    df = pd.read_csv(uploaded_file) if uploaded_file else load_data()
    
    if df is None:
        st.error("데이터를 불러올 수 없습니다.")
        st.stop()
    else:
        st.success("데이터 로드 완료")

# 컬럼 매핑 (공통 사용)
cols = df.columns.tolist()
def find_col(kwd, cs):
    for c in cs:
        if any(k in c.lower() for k in kwd): return c
    return cs[0]

# 생존 여부 컬럼도 찾아야 함 (머신러닝용)
default_age = find_col(['age'], cols)
default_size = find_col(['size', 'tumor'], cols)
default_id = find_col(['id'], cols)
default_survival = find_col(['status', 'survival'], cols) # 생존 여부 (Living/Deceased)

with st.sidebar:
    st.divider()
    st.subheader("🔧 컬럼 매핑")
    col_age = st.selectbox("나이 (Age)", cols, index=cols.index(default_age))
    col_size = st.selectbox("크기 (Size)", cols, index=cols.index(default_size))
    col_surv = st.selectbox("생존여부 (Status)", cols, index=cols.index(default_survival))
    col_id = st.selectbox("ID", cols, index=cols.index(default_id))

# 전처리 (공통)
analysis_df = df.copy()
analysis_df['Age_Clean'] = pd.to_numeric(analysis_df[col_age], errors='coerce')
analysis_df['Size_Clean'] = pd.to_numeric(analysis_df[col_size], errors='coerce')
# 생존 여부 전처리 (Living/Deceased -> 1/0)
# 데이터에 따라 값이 다를 수 있어 가장 흔한 'Living'을 1로 잡음
analysis_df['Surv_Target'] = analysis_df[col_surv].apply(lambda x: 1 if str(x).lower().startswith('l') or str(x) == '1' else 0)

valid_df = analysis_df.dropna(subset=['Age_Clean', 'Size_Clean', 'Surv_Target'])

# ==============================================================================
# 탭 1: 대시보드 (기존 기능)
# ==============================================================================
with tab1:
    st.header("🔍 환자 비교 분석")
    
    if len(valid_df) > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("데이터 수", f"{len(valid_df):,}명")
        c2.metric("평균 나이", f"{valid_df['Age_Clean'].mean():.1f}세")
        c3.metric("생존율", f"{valid_df['Surv_Target'].mean()*100:.1f}%")
        
        st.divider()
        st.subheader("📍 나의 위치")
        
        # 간단 입력
        ic1, ic2 = st.columns(2)
        in_age = ic1.number_input("나이 입력", value=50.0, key='d_age')
        in_size = ic2.number_input("종양 크기 입력", value=25.0, key='d_size')
        
        t1, t2 = st.tabs(["나이 분포", "크기 분포"])
        with t1:
            fig = px.histogram(valid_df, x='Age_Clean', nbins=50, title="나이 분포")
            fig.add_vline(x=in_age, line_color="red", line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            fig = px.histogram(valid_df, x='Size_Clean', nbins=50, title="종양 크기 분포")
            fig.add_vline(x=in_size, line_color="red", line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 탭 2: 머신러닝 (NEW!)
# ==============================================================================
with tab2:
    st.header("🤖 AI 생존 예측 (Machine Learning)")
    st.markdown("""
    과거 데이터를 학습한 **Random Forest AI 모델**이 입력된 조건에 따른 생존 확률을 예측합니다.
    (사용 변수: 나이, 종양 크기)
    """)
    
    if len(valid_df) > 100:
        # 1. 모델 학습
        X = valid_df[['Age_Clean', 'Size_Clean']]
        y = valid_df['Surv_Target']
        
        # 학습/테스트 데이터 분리
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 모델 생성 및 학습
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 정확도 확인
        acc = accuracy_score(y_test, model.predict(X_test))
        st.info(f"💡 현재 AI 모델의 예측 정확도: **{acc*100:.1f}%**")
        
        st.divider()
        
        # 2. 사용자 예측
        col_in, col_res = st.columns([1, 1])
        
        with col_in:
            st.subheader("📋 환자 정보 입력")
            p_age = st.slider("환자 나이", 20, 100, 50)
            p_size = st.slider("종양 크기 (mm)", 0, 200, 20)
            
        with col_res:
            st.subheader("🔮 예측 결과")
            
            # 예측 수행
            prediction = model.predict_proba([[p_age, p_size]])
            survival_prob = prediction[0][1] # 생존(1)일 확률
            
            # 결과 시각화
            if survival_prob >= 0.7:
                color = "green"
                status = "긍정적 (Good)"
            elif survival_prob >= 0.4:
                color = "orange"
                status = "보통 (Moderate)"
            else:
                color = "red"
                status = "위험 (High Risk)"
                
            st.markdown(f"""
            <div class="prediction-card">
                <h3>예상 생존 확률</h3>
                <h1 style="color:{color};">{survival_prob*100:.1f}%</h1>
                <p>예후 판정: <b>{status}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.caption("*주의: 이 결과는 통계적 학습에 의한 추정치이며, 실제 의학적 진단과는 다를 수 있습니다.*")
            
    else:
        st.warning("데이터가 너무 적어 머신러닝을 수행할 수 없습니다.")

# ==============================================================================
# 탭 3: 코드 분석
# ==============================================================================
with tab3:
    st.header("💻 머신러닝 코드 분석")
    st.code("""
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터 준비 (X:문제, y:정답)
X = df[['나이', '종양크기']]
y = df['생존여부']

# 2. 모델 생성 (Random Forest)
# 나무(Tree) 100개를 심어서 투표하는 방식
model = RandomForestClassifier(n_estimators=100)

# 3. 학습 (Training)
model.fit(X, y)

# 4. 예측 (Prediction)
# 새로운 환자 데이터 입력 -> 확률 반환
prob = model.predict_proba([[50세, 20mm]])
    """, language="python")

# ==============================================================================
# 탭 4: 가이드
# ==============================================================================
with tab4:
    st.markdown("### 📚 머신러닝이란?")
    st.markdown("""
    - **입력**: 수천 명의 환자 기록 (나이, 종양크기, 생존여부)
    - **학습**: 컴퓨터가 "나이가 많고 종양이 클수록 위험하구나"라는 패턴을 수학적으로 찾아냄
    - **예측**: 새로운 환자가 왔을 때 그 패턴에 대입하여 결과를 도출함
    """)
