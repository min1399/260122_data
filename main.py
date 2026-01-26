import streamlit as st
import pandas as pd
import plotly.express as px
import kagglehub
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 페이지 설정 ---
st.set_page_config(
    page_title="METABRIC 유방암 AI 분석기",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧬 METABRIC Breast Cancer AI Analysis")
st.caption("유방암 임상 데이터 시각화 및 머신러닝 생존 예측")

# 탭 구성
tab1, tab2, tab3 = st.tabs(["📊 대시보드", "🤖 AI 생존 예측", "데이터 확인(Debug)"])

# --- 데이터 로드 함수 ---
@st.cache_data
def load_data():
    csv_files = glob.glob("*.csv")
    target_csvs = [f for f in csv_files if "METABRIC" in f]
    if target_csvs:
        return pd.read_csv(target_csvs[0], low_memory=False)
    
    try:
        path = kagglehub.dataset_download("gunesevitan/breast-cancer-metabric")
        files = glob.glob(os.path.join(path, "*.csv"))
        target = next((f for f in files if "METABRIC_RNA_Mutation" in f), files[0] if files else None)
        if target:
            return pd.read_csv(target, low_memory=False)
    except:
        pass
    return None

# 사이드바
with st.sidebar:
    st.header("📂 데이터 설정")
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=['csv'])
    df = pd.read_csv(uploaded_file) if uploaded_file else load_data()

if df is None:
    st.error("데이터를 불러올 수 없습니다.")
    st.stop()

# --- 똑똑해진 컬럼 매핑 로직 ---
cols = df.columns.tolist()

def smart_find(keywords, columns):
    # 1단계: 정확히 포함되는 단어 찾기
    for k in keywords:
        for c in columns:
            if k.lower() in c.lower(): return c
    return columns[0]

# 생존 여부는 'Status'가 들어간 컬럼을 우선적으로 찾음
default_age = smart_find(['age'], cols)
default_size = smart_find(['size', 'tumor'], cols)
default_surv = smart_find(['status', 'vital'], cols) # 'Status' 우선 검색
default_id = smart_find(['id', 'patient'], cols)

with st.sidebar:
    st.divider()
    st.subheader("🔧 컬럼 매핑 (확인필수)")
    col_age = st.selectbox("나이 (Age)", cols, index=cols.index(default_age))
    col_size = st.selectbox("크기 (Size)", cols, index=cols.index(default_size))
    col_surv = st.selectbox("생존여부 (Status)", cols, index=cols.index(default_surv))
    col_id = st.selectbox("ID", cols, index=cols.index(default_id))
    
    st.info("Tip: 생존여부는 'Survival Status' 또는 'Vital Status'를 선택하세요.")

# --- 전처리 ---
analysis_df = df.copy()
analysis_df['Age_Clean'] = pd.to_numeric(analysis_df[col_age], errors='coerce')
analysis_df['Size_Clean'] = pd.to_numeric(analysis_df[col_size], errors='coerce')

# 생존 여부 타겟팅 (Living/Deceased 또는 0/1)
# 문자열(Living 등)이면 1, 0으로 변환
def parse_survival(val):
    s = str(val).lower()
    if 'liv' in s or s == '1': return 1 # Living, Alive
    if 'die' in s or 'dec' in s or s == '0': return 0 # Died, Deceased
    return None # 모를 경우

analysis_df['Surv_Target'] = analysis_df[col_surv].apply(parse_survival)

# 결측치 제거
valid_df = analysis_df.dropna(subset=['Age_Clean', 'Size_Clean', 'Surv_Target'])

# ==============================================================================
# 탭 1: 대시보드
# ==============================================================================
with tab1:
    st.header("🔍 데이터 시각화")
    if len(valid_df) > 0:
        c1, c2 = st.columns(2)
        c1.metric("분석 데이터 수", f"{len(valid_df):,}명")
        c2.metric("평균 생존율", f"{valid_df['Surv_Target'].mean()*100:.1f}%")
        
        fig = px.scatter(valid_df, x='Age_Clean', y='Size_Clean', color=valid_df['Surv_Target'].astype(str),
                         title="나이 vs 종양크기 생존 분포", opacity=0.6,
                         labels={'color': '생존여부(1=생존)'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("유효한 데이터가 0개입니다. 사이드바의 컬럼 매핑을 확인해주세요.")

# ==============================================================================
# 탭 2: AI 생존 예측
# ==============================================================================
with tab2:
    st.header("🤖 AI 생존 예측")
    
    if len(valid_df) > 50:
        # 모델 학습
        X = valid_df[['Age_Clean', 'Size_Clean']]
        y = valid_df['Surv_Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        st.success(f"AI 모델 학습 완료! (정확도: {acc*100:.1f}%)")
        
        # 예측기
        st.subheader("생존 확률 예측해보기")
        c1, c2 = st.columns(2)
        in_age = c1.slider("환자 나이", 20, 100, 50)
        in_size = c2.slider("종양 크기 (mm)", 0, 200, 20)
        
        pred = model.predict_proba([[in_age, in_size]])
        prob = pred[0][1] * 100 # 생존 확률
        
        st.metric(label="예상 생존 확률", value=f"{prob:.1f}%")
        
        if prob > 70:
            st.success("비교적 긍정적인 예후가 예상됩니다.")
        elif prob > 40:
            st.warning("주의가 필요한 단계입니다.")
        else:
            st.error("높은 위험도가 예상됩니다.")
            
    else:
        st.warning("데이터가 부족합니다. (탭3에서 데이터를 확인하세요)")

# ==============================================================================
# 탭 3: 디버깅 (문제 해결용)
# ==============================================================================
# ==============================================================================
# 탭 3: 디버깅 (수정된 버전)
# ==============================================================================
with tab3:
    st.header("🛠 데이터가 왜 없지?")
    st.write("현재 선택된 컬럼의 데이터 상태를 보여줍니다.")
    
    # 수정 포인트: .values 뒤에 .tolist()를 붙여서 일반 리스트로 변환
    
    st.write(f"1. **나이 컬럼 ({col_age})** 샘플:")
    # st.write(df[col_age].head(3).values)  <-- 기존 코드 (에러 원인)
    st.write(df[col_age].head(3).tolist()) # <-- 수정 코드 (안전함)
    
    st.write(f"2. **크기 컬럼 ({col_size})** 샘플:")
    st.write(df[col_size].head(3).tolist()) 
    
    st.write(f"3. **생존 컬럼 ({col_surv})** 샘플:")
    st.write(df[col_surv].head(3).tolist())
    
    st.divider()
    
    st.write("변환 후 데이터 (상위 5개):")
    # dataframe은 st.dataframe으로 보여주는 게 가장 안전합니다.
    st.dataframe(valid_df.head())
    st.write("변환 후 데이터 (상위 5개):")
    st.dataframe(valid_df.head())
