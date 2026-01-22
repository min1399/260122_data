import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="데이터 분석 대시보드", layout="wide")

# 한글 폰트 지원을 위한 설정 (필요 시 운영체제에 맞춰 추가 설정 가능)
st.title("📊 데이터 분석 및 시각화 앱")

# 1. 데이터 로드 함수
@st.cache_data
def load_data(file):
    try:
        # csv 파일 읽기 (한글 인코딩 대응)
        try:
            df = pd.read_csv(file, encoding='cp949')
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return None

# 사이드바: 파일 업로드
st.sidebar.header("데이터 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=['csv'])

# 기본 파일 설정 (업로드된 파일이 없을 경우 기본 파일 사용)
# 로컬 테스트 시 같은 폴더에 해당 파일이 있어야 합니다.
default_file_name = "202512_202512____________________________.csv"
df = None

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.sidebar.success("업로드된 파일을 사용합니다.")
else:
    # 기본 파일 로드 시도
    try:
        df = load_data(default_file_name)
        st.sidebar.info(f"기본 파일({default_file_name})을 로드했습니다.")
    except FileNotFoundError:
        st.warning("기본 데이터 파일을 찾을 수 없습니다. CSV 파일을 업로드해주세요.")

# 데이터가 로드되었을 때 실행
if df is not None:
    # 2. 데이터 컬럼 분석을 통해 모드 결정
    cols = df.columns.tolist()
    
    # (A) 인구 데이터 판별 (업로드하신 파일 형태)
    is_population_data = any("연령구간인구수" in col for col in cols) or any("0세" in col for col in cols)
    
    # (B) 날씨 데이터 판별 (기온 비교 요청용)
    # 예: '날짜' 또는 '일시', '평균기온' 컬럼이 있는지 확인
    weather_date_col = next((c for c in cols if "날짜" in c or "일시" in c), None)
    weather_temp_col = next((c for c in cols if "기온" in c or "temperature" in c.lower()), None)
    is_weather_data = weather_date_col is not None and weather_temp_col is not None

    # --- 화면 분기 ---
    
    if is_population_data:
        st.subheader("👥 인구 구조 분석 모드")
        st.info("업로드된 데이터가 **인구 통계 데이터**로 인식되었습니다.")
        
        # 전처리: 콤마 제거 및 숫자 변환
        # 행정구역 컬럼 찾기
        region_col = cols[0] # 보통 첫 번째 컬럼이 행정구역
        
        # 지역 선택
        region_list = df[region_col].unique()
        selected_region = st.selectbox("분석할 행정구역을 선택하세요", region_list)
        
        # 선택된 지역 데이터 필터링
        region_data = df[df[region_col] == selected_region].iloc[0]
        
        # 연령 데이터 추출 ('0세' 부터 끝까지 혹은 '100세 이상'까지)
        # 컬럼명에 '세'가 포함된 컬럼만 추출
        age_cols = [c for c in cols if '세' in c and '연령구간' not in c and '총인구수' not in c]
        
        if age_cols:
            # 데이터 정제 (문자열 숫자의 콤마 제거)
            age_values = []
            valid_age_cols = []
            for c in age_cols:
                val = str(region_data[c])
                val = val.replace(',', '')
                if val.isdigit():
                    age_values.append(int(val))
                    valid_age_cols.append(c)
            
            # 그래프 그리기
            chart_df = pd.DataFrame({'연령': valid_age_cols, '인구수': age_values})
            
            # Plotly Bar Chart
            fig = px.bar(chart_df, x='연령', y='인구수', 
                         title=f"{selected_region} 연령별 인구 분포",
                         labels={'인구수': '인구 수(명)', '연령': '나이'})
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(chart_df.T)
        else:
            st.error("연령 컬럼을 찾을 수 없습니다.")

    elif is_weather_data:
        st.subheader("🌡️ 날씨 기온 비교 모드")
        st.info("업로드된 데이터가 **기온 데이터**로 인식되었습니다.")
        
        # 날짜 컬럼을 datetime으로 변환
        df[weather_date_col] = pd.to_datetime(df[weather_date_col])
        df = df.sort_values(by=weather_date_col)
        
        # 기준 날짜 선택 (기본값: 데이터의 가장 최근 날짜)
        max_date = df[weather_date_col].max()
        min_date = df[weather_date_col].min()
        
        st.write(f"데이터 기간: {min_date.date()} ~ {max_date.date()}")
        
        target_date = st.date_input("비교할 기준 날짜를 선택하세요", value=max_date.date(), 
                                    min_value=min_date.date(), max_value=max_date.date())
        
        # 선택한 날짜의 데이터 확인
        target_row = df[df[weather_date_col].dt.date == target_date]
        
        if not target_row.empty:
            target_temp = target_row[weather_temp_col].values[0]
            st.metric(label=f"{target_date}의 기온", value=f"{target_temp}℃")
            
            # 과거의 같은 날짜(월, 일) 데이터 찾기
            target_month = target_date.month
            target_day = target_date.day
            
            history_df = df[(df[weather_date_col].dt.month == target_month) & 
                            (df[weather_date_col].dt.day == target_day) &
                            (df[weather_date_col].dt.date != target_date)].copy()
            
            if not history_df.empty:
                # 비교 로직
                history_df['기온차'] = target_temp - history_df[weather_temp_col]
                
                # Plotly 시각화
                # 1. 과거 같은 날짜들의 기온 추세선
                fig_trend = px.line(history_df, x=weather_date_col, y=weather_temp_col, markers=True,
                                    title=f"과거 {target_month}월 {target_day}일의 기온 변화")
                
                # 기준 날짜 기온 점선 추가
                fig_trend.add_hline(y=target_temp, line_dash="dash", line_color="red", 
                                    annotation_text="기준일 기온", annotation_position="top left")
                
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # 2. 얼마나 더웠나/추웠나 비교 텍스트
                avg_past_temp = history_df[weather_temp_col].mean()
                diff = target_temp - avg_past_temp
                
                status = "더움" if diff > 0 else "추움"
                st.write(f"### 분석 결과")
                st.write(f"선택하신 **{target_date}**은 과거 같은 날짜들의 평균 기온({avg_past_temp:.1f}℃) 대비 **약 {abs(diff):.1f}℃ {status}**.")
                
                st.write("#### 과거 기록 상세")
                st.dataframe(history_df[[weather_date_col, weather_temp_col, '기온차']].sort_values(by=weather_date_col, ascending=False))
                
            else:
                st.warning("과거의 같은 날짜 데이터를 찾을 수 없습니다.")
        else:
            st.error("선택한 날짜의 데이터가 없습니다.")

    else:
        st.warning("데이터 형식을 인식할 수 없습니다. '행정구역/연령' 또는 '날짜/기온' 컬럼이 포함된 CSV를 업로드해주세요.")
        st.write("현재 로드된 데이터 컬럼:", cols)
        st.dataframe(df.head())
