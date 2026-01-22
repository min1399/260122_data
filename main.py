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
    # 1. 로컬에 있는 CSV 파일 먼저 탐색 (사용자가 깃허브에 올린 파일)
    # 현재 폴더의 모든 csv 파일을 찾습니다.
    local_csvs = glob.glob("*.csv")
    
    # 인구 데이터 파일은 제외 (파일명으로 필터링)
    target_csvs = [f for f in local_csvs if "202512" not in f]
    
    if target_csvs:
        # 가장 첫 번째 발견된 csv를 사용
        file_path = target_csvs[0]
        st.toast(f"로컬 파일 발견: {file_path}")
        return pd.read_csv(file_path, low_memory=False)
    
    # 2. 로컬 파일이 없으면 Kaggle 다운로드 시도
    try:
        st.toast("로컬 파일이 없어 Kaggle에서 다운로드를 시도합니다...")
        path = kagglehub.dataset_download("gunesevitan/breast-cancer-metabric")
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        
        target_file = None
        for f in csv_files:
            if "METABRIC_RNA_Mutation" in f:
                target_file = f
                break
        if not target_file and csv_files:
            target_file = csv_files[0]
            
        if target_file:
            return pd.read_csv(target_file, low_memory=False)
    except Exception as e:
        return None
    
    return None

# 사이드바 설정
st.sidebar.header("데이터 설정")
uploaded_file = st.sidebar.file_uploader("CSV 데이터 업로드 (선택)", type=['csv'])

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("업로드된 파일을 사용합니다.")
    except:
        st.sidebar.error("파일을 읽을 수 없습니다.")
else:
    df = load_data()
    if df is not None:
        st.sidebar.info("기본/로컬 데이터셋을 사용합니다.")
    else:
        st.warning("데이터를 찾을 수 없습니다. CSV 파일을 업로드하거나 깃허브에 파일을 추가해주세요.")
        st.stop()

# --- 여기서부터가 핵심 수정 부분입니다 (컬럼 매핑) ---

if df is not None:
    st.write("### 📂 데이터 미리보기 (상위 3행)")
    st.dataframe(df.head(3))
    
    # 컬럼 이름이 제각각일 수 있으므로, 사용자가 선택하게 하거나 자동 감지 시도
    all_columns = df.columns.tolist()
    
    # 자동 감지 로직 (대소문자 무시하고 키워드 찾기)
    def find_col(keywords):
        for col in all_columns:
            if any(k in col.lower() for k in keywords):
                return col
        return None

    # 기본값 추정
    default_age = find_col(['age', 'diagnosis'])
    default_size = find_col(['size', 'tumor', 'diameter'])
    default_survival = find_col(['survival', 'month', 'os'])
    default_id = find_col(['id', 'patient'])

    st.sidebar.subheader("🔧 컬럼 매핑 (자동 감지됨)")
    
    # 만약 자동 감지가 틀렸다면 사용자가 바꿀 수 있게 selectbox 제공
    col_age = st.sidebar.selectbox("나이(Age) 컬럼", all_columns, index=all_columns.index(default_age) if default_age else 0)
    col_size = st.sidebar.selectbox("종양 크기(Size) 컬럼", all_columns, index=all_columns.index(default_size) if default_size else 0)
    col_survival = st.sidebar.selectbox("생존 기간(Survival) 컬럼 (선택)", [None] + all_columns, index=all_columns.index(default_survival) + 1 if default_survival else 0)
    col_id = st.sidebar.selectbox("환자 ID 컬럼", all_columns, index=all_columns.index(default_id) if default_id else 0)

    # 필수 컬럼 데이터 확인
    if col_age and col_size:
        # 결측치 제거
        analysis_df = df.dropna(subset=[col_age, col_size]).copy()
        
        # 데이터 타입 변환 (숫자로)
        analysis_df[col_age] = pd.to_numeric(analysis_df[col_age], errors='coerce')
        analysis_df[col_size] = pd.to_numeric(analysis_df[col_size], errors='coerce')
        analysis_df = analysis_df.dropna(subset=[col_age, col_size])

        # --- 메인 기능 시작 ---
        st.divider()
        st.header("1. 데이터 요약")
        c1, c2, c3 = st.columns(3)
        c1.metric("분석 대상 환자 수", f"{len(analysis_df):,}명")
        c2.metric("평균 나이", f"{analysis_df[col_age].mean():.1f}세")
        c3.metric("평균 종양 크기", f"{analysis_df[col_size].mean():.1f}mm")

        # --- 비교 분석 모드 ---
        st.header("2. 나의 위치 확인 (Compare)")
        
        # 입력 방식
        input_type = st.radio("입력 방식", ["ID로 찾기", "직접 입력"], horizontal=True)
        
        my_age = 0.0
        my_size = 0.0
        
        if input_type == "ID로 찾기":
            # ID 검색
            patient_list = analysis_df[col_id].astype(str).unique()
            selected_id = st.selectbox("환자 ID 선택", patient_list)
            
            row = analysis_df[analysis_df[col_id].astype(str) == selected_id].iloc[0]
            my_age = row[col_age]
            my_size = row[col_size]
            st.success(f"선택한 환자: 나이 {my_age}세, 크기 {my_size}mm")
            
        else:
            c1, c2 = st.columns(2)
            my_age = c1.number_input("나이 입력", value=50.0)
            my_size = c2.number_input("종양 크기 입력", value=25.0)

        # 시각화 (Plotly)
        st.subheader("📊 분포 상 나의 위치")
        
        # 탭으로 분리
        tab1, tab2 = st.tabs(["나이 분포", "종양 크기 분포"])
        
        with tab1:
            fig_age = px.histogram(analysis_df, x=col_age, nbins=50, title="나이 분포")
            fig_age.add_vline(x=my_age, line_dash="dash", line_color="red", annotation_text="나")
            st.plotly_chart(fig_age, use_container_width=True)
            
            # 백분위 계산
            percentile_age = (analysis_df[col_age] < my_age).mean() * 100
            st.caption(f"당신의 나이는 하위 {percentile_age:.1f}% (상위 {100-percentile_age:.1f}%)에 해당합니다.")

        with tab2:
            fig_size = px.histogram(analysis_df, x=col_size, nbins=50, title="종양 크기 분포")
            fig_size.add_vline(x=my_size, line_dash="dash", line_color="red", annotation_text="나")
            st.plotly_chart(fig_size, use_container_width=True)
            
            percentile_size = (analysis_df[col_size] < my_size).mean() * 100
            st.caption(f"당신의 종양 크기는 하위 {percentile_size:.1f}% (상위 {100-percentile_size:.1f}%)에 해당합니다.")
            
    else:
        st.error("사이드바에서 정확한 컬럼을 선택해주세요.")
