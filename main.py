import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import kagglehub
import os
import glob

# 페이지 설정
st.set_page_config(page_title="METABRIC 유방암 데이터 분석기", layout="wide")

st.title("🧬 METABRIC Breast Cancer Data Analysis")

# 1. 데이터 로드 함수
@st.cache_data
def load_data():
    # 로컬 CSV 탐색 (Breast Cancer 파일 우선)
    csv_files = glob.glob("*.csv")
    
    # 1순위: 파일명에 'METABRIC'이 포함된 것
    target_csvs = [f for f in csv_files if "METABRIC" in f]
    
    if not target_csvs:
        # 2순위: 202512가 없는 다른 csv
        target_csvs = [f for f in csv_files if "202512" not in f]
    
    if target_csvs:
        file_path = target_csvs[0]
        st.toast(f"로컬 파일 발견: {file_path}")
        return pd.read_csv(file_path, low_memory=False)
    
    # 3순위: Kaggle 다운로드
    try:
        st.toast("데이터 다운로드 중...")
        path = kagglehub.dataset_download("gunesevitan/breast-cancer-metabric")
        files = glob.glob(os.path.join(path, "*.csv"))
        target = next((f for f in files if "METABRIC_RNA_Mutation" in f), files[0] if files else None)
        if target:
            return pd.read_csv(target, low_memory=False)
    except Exception:
        pass
    
    return None

# 사이드바 설정
st.sidebar.header("📂 데이터 설정")
uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=['csv'])

df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("업로드 파일 사용")
    except Exception as e:
        st.error(f"파일 읽기 실패: {e}")
else:
    df = load_data()

if df is None:
    st.warning("데이터가 없습니다.")
    st.stop()

# --- 컬럼 자동 매핑 (핵심 수정) ---

cols = df.columns.tolist()

# 우선순위가 높은 컬럼명을 먼저 찾도록 정의
def find_column(candidates, columns):
    for candidate in candidates:
        for col in columns:
            if candidate.lower() == col.lower(): # 정확히 일치
                return col
            if candidate.lower() in col.lower(): # 포함됨
                return col
    return columns[0] # 못 찾으면 첫 번째 컬럼

# 실제 데이터셋의 컬럼명 반영
default_age_col = find_column(['Age at Diagnosis', 'age'], cols)
default_size_col = find_column(['Tumor Size', 'size'], cols)
default_id_col = find_column(['Patient ID', 'id'], cols)

st.sidebar.subheader("🔧 컬럼 매핑 확인")
col_age = st.sidebar.selectbox("나이(Age)", cols, index=cols.index(default_age_col))
col_size = st.sidebar.selectbox("종양크기(Size)", cols, index=cols.index(default_size_col))
col_id = st.sidebar.selectbox("환자ID", cols, index=cols.index(default_id_col))

# --- 데이터 전처리 (안전하게 처리) ---

# 원본 데이터 보존을 위해 복사
analysis_df = df.copy()

# 숫자 변환 (변환할 수 없는 값은 NaN으로 처리됨)
analysis_df['Analyze_Age'] = pd.to_numeric(analysis_df[col_age], errors='coerce')
analysis_df['Analyze_Size'] = pd.to_numeric(analysis_df[col_size], errors='coerce')

# NaN 제거 (유효한 데이터만 남김)
valid_data = analysis_df.dropna(subset=['Analyze_Age', 'Analyze_Size'])

# 유효 데이터 개수 확인
if len(valid_data) == 0:
    st.error("🚨 오류: 유효한 데이터가 0개입니다.")
    st.write(f"선택된 컬럼: {col_age}, {col_size}")
    st.write("원본 데이터 샘플:")
    st.dataframe(df[[col_age, col_size]].head())
    st.stop()
else:
    analysis_df = valid_data

# --- 메인 대시보드 ---

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("분석 환자 수", f"{len(analysis_df):,}명")
c2.metric("평균 나이", f"{analysis_df['Analyze_Age'].mean():.1f}세")
c3.metric("평균 종양 크기", f"{analysis_df['Analyze_Size'].mean():.1f}mm")

st.header("🔍 나의 위치 분석")
input_type = st.radio("입력 방식", ["ID로 찾기", "직접 입력"], horizontal=True)

my_age, my_size = 0.0, 0.0
valid_input = False

if input_type == "ID로 찾기":
    # ID 검색 (문자열로 변환하여 비교)
    analysis_df[col_id] = analysis_df[col_id].astype(str)
    patient_list = analysis_df[col_id].unique()
    
    if len(patient_list) > 0:
        selected_id = st.selectbox("환자 ID 선택", patient_list)
        target_row = analysis_df[analysis_df[col_id] == selected_id]
        
        if not target_row.empty:
            row = target_row.iloc[0]
            my_age = row['Analyze_Age']
            my_size = row['Analyze_Size']
            st.success(f"ID {selected_id}: 나이 {my_age:.1f}세, 크기 {my_size:.1f}mm")
            valid_input = True
    else:
        st.warning("ID 컬럼에 유효한 데이터가 없습니다.")

else: # 직접 입력
    c1, c2 = st.columns(2)
    my_age = c1.number_input("나이 입력", value=50.0)
    my_size = c2.number_input("종양 크기 입력", value=25.0)
    valid_input = True

if valid_input:
    tab1, tab2 = st.tabs(["📊 나이 분포", "📉 종양 크기 분포"])
    
    with tab1:
        fig = px.histogram(analysis_df, x='Analyze_Age', nbins=50, title="나이 분포")
        fig.add_vline(x=my_age, line_dash="dash", line_color="red", annotation_text="나")
        st.plotly_chart(fig, use_container_width=True)
        
        pct = (analysis_df['Analyze_Age'] < my_age).mean() * 100
        st.caption(f"당신은 상위 {100-pct:.1f}% (하위 {pct:.1f}%) 연령대에 속합니다.")

    with tab2:
        fig = px.histogram(analysis_df, x='Analyze_Size', nbins=50, title="종양 크기 분포")
        fig.add_vline(x=my_size, line_dash="dash", line_color="red", annotation_text="나")
        st.plotly_chart(fig, use_container_width=True)
        
        pct = (analysis_df['Analyze_Size'] < my_size).mean() * 100
        st.caption(f"당신은 상위 {100-pct:.1f}% (하위 {pct:.1f}%) 크기에 속합니다.")
