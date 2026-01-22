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

# 1. 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def load_data():
    # 1. 사용자가 업로드한 파일이 있는지 확인 (Streamlit UI를 통해)
    # (이 함수 내부에서는 file_uploader의 결과를 직접 받을 수 없으므로, 외부에서 처리 후 넘겨받는 구조가 좋으나,
    #  여기서는 '기본 탑재' 로직을 위해 Kaggle 다운로드를 우선 구현합니다.)
    
    try:
        # Kagglehub를 통해 데이터 다운로드 (최초 1회 실행 후 캐시됨)
        # 로컬에 파일이 없다면 다운로드 시도
        st.toast("Kaggle에서 데이터셋을 확인 중입니다...")
        path = kagglehub.dataset_download("gunesevitan/breast-cancer-metabric")
        
        # 다운로드된 폴더 내의 CSV 파일 찾기
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        
        # 보통 'METABRIC_RNA_Mutation.csv'가 메인 데이터입니다.
        target_file = None
        for f in csv_files:
            if "METABRIC_RNA_Mutation" in f:
                target_file = f
                break
        
        if target_file is None and csv_files:
            target_file = csv_files[0]
            
        if target_file:
            # low_memory=False는 컬럼 타입 추론 경고 방지
            df = pd.read_csv(target_file, low_memory=False)
            return df
        else:
            return None
            
    except Exception as e:
        st.error(f"데이터 다운로드 중 오류 발생: {e}")
        return None

# 사이드바: 데이터 업로드 및 설정
st.sidebar.header("데이터 설정")
uploaded_file = st.sidebar.file_uploader("새로운 CSV 데이터 업로드 (선택)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("업로드된 파일을 사용합니다.")
else:
    df = load_data()
    if df is not None:
        st.sidebar.info("기본 METABRIC 데이터셋을 사용합니다.")
    else:
        st.stop() # 데이터가 없으면 중단

# 데이터 전처리 (주요 컬럼 정리)
# METABRIC 데이터셋의 주요 컬럼명 매핑 (데이터셋 버전에 따라 다를 수 있어 확인 필요)
# 주요 컬럼: age_at_diagnosis, tumor_size, overall_survival_months, cellularity 등
required_cols = ['age_at_diagnosis', 'tumor_size', 'overall_survival_months', 'patient_id']
available_cols = [c for c in required_cols if c in df.columns]

if len(available_cols) < 2:
    st.error("데이터셋에서 필요한 컬럼(age, tumor_size 등)을 찾을 수 없습니다.")
    st.write("현재 컬럼 목록:", df.columns.tolist())
else:
    # 결측치 제거 (분석용)
    analysis_df = df.dropna(subset=['age_at_diagnosis', 'tumor_size'])

    # --- 메인 기능 ---
    
    st.header("1. 전체 데이터 요약")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("총 환자 수", f"{len(analysis_df):,}명")
    c2.metric("평균 진단 나이", f"{analysis_df['age_at_diagnosis'].mean():.1f}세")
    c3.metric("평균 종양 크기", f"{analysis_df['tumor_size'].mean():.1f}mm")

    # --- 비교 분석 모드 (이전의 '날씨 비교' 기능 대체) ---
    st.header("2. 환자 비교 분석 (Interactive)")
    st.info("특정 환자를 선택하거나 수치를 입력하여 전체 환자군과 비교합니다.")

    # 입력 방식 선택
    input_mode = st.radio("비교 대상 선택", ["기존 환자 ID로 검색", "가상 데이터 직접 입력"], horizontal=True)
    
    target_data = {}
    
    if input_mode == "기존 환자 ID로 검색":
        # 환자 ID 선택
        patient_ids = analysis_df['patient_id'].astype(str).tolist()
        selected_id = st.selectbox("환자 ID 선택", patient_ids)
        
        # 선택된 환자 정보 추출
        # patient_id가 int일 수도, str일 수도 있으므로 매칭 주의
        selected_row = analysis_df[analysis_df['patient_id'].astype(str) == selected_id].iloc[0]
        
        target_data = {
            'age': selected_row['age_at_diagnosis'],
            'size': selected_row['tumor_size'],
            'survival': selected_row.get('overall_survival_months', 0)
        }
        st.write(f"**선택된 환자({selected_id}) 정보:** 나이 {target_data['age']:.1f}세, 종양크기 {target_data['size']:.1f}mm")
        
    else:
        c1, c2 = st.columns(2)
        input_age = c1.number_input("나이 (Age)", value=50.0, step=0.5)
        input_size = c2.number_input("종양 크기 (Tumor Size, mm)", value=20.0, step=1.0)
        target_data = {'age': input_age, 'size': input_size}

    # --- 시각화 및 비교 로직 ---
    
    # 1. 나이 비교
    st.subheader("📊 진단 나이 비교")
    mean_age = analysis_df['age_at_diagnosis'].mean()
    diff_age = target_data['age'] - mean_age
    status_age = "많음" if diff_age > 0 else "적음"
    
    st.markdown(f"""
    선택 대상의 나이는 **{target_data['age']:.1f}세**로, 전체 평균({mean_age:.1f}세)보다 
    **약 {abs(diff_age):.1f}세 {status_age}** (상위 {len(analysis_df[analysis_df['age_at_diagnosis'] > target_data['age']]) / len(analysis_df) * 100:.1f}% 구간).
    """)
    
    # 히스토그램 + 수직선
    fig_age = px.histogram(analysis_df, x='age_at_diagnosis', nbins=50, title="전체 환자 나이 분포")
    fig_age.add_vline(x=target_data['age'], line_width=3, line_dash="dash", line_color="red", annotation_text="선택 대상")
    st.plotly_chart(fig_age, use_container_width=True)

    # 2. 종양 크기 비교
    st.subheader("📊 종양 크기 비교")
    mean_size = analysis_df['tumor_size'].mean()
    diff_size = target_data['size'] - mean_size
    status_size = "큼" if diff_size > 0 else "작음"
    
    st.markdown(f"""
    선택 대상의 종양 크기는 **{target_data['size']:.1f}mm**로, 전체 평균({mean_size:.1f}mm)보다 
    **약 {abs(diff_size):.1f}mm {status_size}**.
    """)
    
    # Box Plot + 점 표시
    fig_size = px.box(analysis_df, x='tumor_size', title="전체 환자 종양 크기 분포 (Box Plot)")
    # Scatter로 점 찍기 (y는 boxplot 위치에 맞춤, 보통 0 근처)
    fig_size.add_trace(go.Scatter(x=[target_data['size']], y=[0], mode='markers', 
                                  marker=dict(color='red', size=15, symbol='diamond'), 
                                  name='선택 대상'))
    st.plotly_chart(fig_size, use_container_width=True)

    # 3. 산점도 (나이 vs 종양크기) 내 위치 확인
    st.subheader("📍 전체 환자군 내 위치 확인")
    fig_scatter = px.scatter(analysis_df, x='age_at_diagnosis', y='tumor_size', 
                             color='overall_survival_months', opacity=0.5,
                             title="나이 vs 종양 크기 분포 (색상: 생존 기간)")
    
    fig_scatter.add_trace(go.Scatter(x=[target_data['age']], y=[target_data['size']],
                                     mode='markers+text',
                                     marker=dict(color='red', size=20, symbol='x'),
                                     text=["HERE"], textposition="top center",
                                     name='선택 대상'))
    
    st.plotly_chart(fig_scatter, use_container_width=True)
