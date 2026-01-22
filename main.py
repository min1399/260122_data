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

# 1. 데이터 로드 함수 (안전성 강화)
@st.cache_data
def load_data():
    # 1. 로컬 CSV 탐색 (파일명에 202512가 없는 파일 우선)
    csv_files = glob.glob("*.csv")
    target_csvs = [f for f in csv_files if "202512" not in f]
    
    if target_csvs:
        st.toast(f"로컬 파일 발견: {target_csvs[0]}")
        return pd.read_csv(target_csvs[0], low_memory=False)
    
    # 2. Kaggle 데이터 다운로드
    try:
        st.toast("Kaggle에서 데이터 다운로드 중...")
        path = kagglehub.dataset_download("gunesevitan/breast-cancer-metabric")
        files = glob.glob(os.path.join(path, "*.csv"))
        
        target = next((f for f in files if "METABRIC_RNA_Mutation" in f), files[0] if files else None)
        
        if target:
            return pd.read_csv(target, low_memory=False)
    except Exception:
        pass
    
    return None

# 사이드바: 데이터 설정
st.sidebar.header("📂 데이터 설정")
uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=['csv'])

df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("업로드 파일 사용 중")
    except Exception as e:
        st.sidebar.error(f"파일 읽기 실패: {e}")
else:
    df = load_data()
    if df is not None:
        st.sidebar.info("기본/로컬 데이터 사용 중")
    else:
        st.warning("분석할 데이터가 없습니다. CSV 파일을 업로드해주세요.")
        st.stop()

# --- 데이터 컬럼 매핑 및 전처리 ---

if df is not None:
    # 컬럼 자동 감지 함수
    cols = df.columns.tolist()
    def get_idx(keywords):
        for i, col in enumerate(cols):
            if any(k in col.lower() for k in keywords):
                return i
        return 0

    st.sidebar.subheader("🔧 컬럼 매핑")
    col_age = st.sidebar.selectbox("나이(Age)", cols, index=get_idx(['age', 'diagnosis']))
    col_size = st.sidebar.selectbox("종양크기(Size)", cols, index=get_idx(['size', 'tumor']))
    col_id = st.sidebar.selectbox("환자ID", cols, index=get_idx(['id', 'patient']))

    # 데이터 전처리 (숫자 변환 및 결측치 제거)
    analysis_df = df.copy()
    
    # 숫자로 변환 (에러 발생 시 NaN 처리)
    analysis_df[col_age] = pd.to_numeric(analysis_df[col_age], errors='coerce')
    analysis_df[col_size] = pd.to_numeric(analysis_df[col_size], errors='coerce')
    
    # NaN이 있는 행 제거
    analysis_df = analysis_df.dropna(subset=[col_age, col_size])

    # [중요] 전처리 후 데이터가 비어있는지 체크 (IndexError 방지)
    if len(analysis_df) == 0:
        st.error("🚨 오류: 유효한 데이터가 없습니다!")
        st.markdown("""
        **가능한 원인:**
        1. 선택한 컬럼(`나이`, `종양크기`)에 숫자가 아닌 데이터(문자 등)가 들어있어서 모두 삭제되었습니다.
        2. 사이드바의 **'컬럼 매핑'**이 올바른지 확인해주세요. (예: 행정구역 컬럼을 나이로 선택하지 않았나요?)
        """)
        st.stop()

    # --- 메인 대시보드 ---
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("분석 환자 수", f"{len(analysis_df):,}명")
    c2.metric("평균 나이", f"{analysis_df[col_age].mean():.1f}세")
    c3.metric("평균 종양 크기", f"{analysis_df[col_size].mean():.1f}mm")

    st.header("🔍 나의 위치 분석")
    input_type = st.radio("입력 방식", ["ID로 찾기", "직접 입력"], horizontal=True)

    my_age, my_size = 0.0, 0.0
    valid_input = False

    if input_type == "ID로 찾기":
        # ID 리스트 생성
        patient_list = analysis_df[col_id].astype(str).unique()
        
        if len(patient_list) > 0:
            selected_id = st.selectbox("환자 ID 선택", patient_list)
            
            # 선택된 ID 데이터 필터링
            target_row = analysis_df[analysis_df[col_id].astype(str) == selected_id]
            
            if not target_row.empty:
                # [수정됨] 안전하게 값 가져오기
                row = target_row.iloc[0]
                my_age = row[col_age]
                my_size = row[col_size]
                st.success(f"ID {selected_id}: 나이 {my_age}세, 크기 {my_size}mm")
                valid_input = True
            else:
                st.error("해당 ID의 데이터를 찾을 수 없습니다.")
        else:
            st.error("표시할 환자 ID가 없습니다.")
            
    else: # 직접 입력
        c1, c2 = st.columns(2)
        my_age = c1.number_input("나이 입력", value=50.0)
        my_size = c2.number_input("종양 크기 입력", value=25.0)
        valid_input = True

    # 그래프 그리기
    if valid_input:
        tab1, tab2 = st.tabs(["📊 나이 분포", "📉 종양 크기 분포"])
        
        with tab1:
            fig = px.histogram(analysis_df, x=col_age, nbins=50, title="나이 분포")
            fig.add_vline(x=my_age, line_dash="dash", line_color="red", annotation_text="나")
            st.plotly_chart(fig, use_container_width=True)
            
            pct = (analysis_df[col_age] < my_age).mean() * 100
            st.caption(f"당신은 상위 {100-pct:.1f}% (하위 {pct:.1f}%) 연령대에 속합니다.")

        with tab2:
            fig = px.histogram(analysis_df, x=col_size, nbins=50, title="종양 크기 분포")
            fig.add_vline(x=my_size, line_dash="dash", line_color="red", annotation_text="나")
            st.plotly_chart(fig, use_container_width=True)
            
            pct = (analysis_df[col_size] < my_size).mean() * 100
            st.caption(f"당신은 상위 {100-pct:.1f}% (하위 {pct:.1f}%) 크기에 속합니다.")
