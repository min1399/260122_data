# 🧬 METABRIC Breast Cancer Analysis App

이 프로젝트는 Kaggle의 [METABRIC Breast Cancer Dataset](https://www.kaggle.com/datasets/gunesevitan/breast-cancer-metabric)을 활용하여 구축된 Streamlit 웹 애플리케이션입니다.
임상 데이터(Clinical Data)를 기반으로 **특정 환자의 상태가 전체 환자군 내에서 어떤 위치에 있는지 비교 분석**하는 기능을 제공합니다.

## 🚀 주요 기능 (Features)

### 1. 데이터 자동 연동 (Auto Data Fetch)
- `kagglehub` 라이브러리를 사용하여 앱 실행 시 최신 데이터셋을 자동으로 다운로드합니다.
- 별도의 CSV 다운로드 없이도 즉시 분석이 가능합니다.
- 사용자가 별도의 유사 형식 CSV를 업로드할 경우 해당 데이터를 우선 분석합니다.

### 2. 환자 비교 분석 (Patient Comparison)
날씨 앱의 "평년 기온 비교" 기능을 임상 데이터에 맞게 재해석했습니다.
- **Input:** 특정 환자 ID를 선택하거나, 나이/종양 크기 등의 수치를 직접 입력합니다.
- **Analysis:** 입력된 수치가 전체 환자 분포(Distribution)에서 어디에 위치하는지 계산합니다.
  - 예: *"선택 환자의 종양 크기는 평균 대비 5mm 큽니다."*
- **Visualization:** 히스토그램과 Box Plot 위에 사용자의 위치를 붉은 점/선으로 표시하여 직관적으로 보여줍니다.

## 🛠 설치 및 실행 (Installation & Usage)

### 1. 패키지 설치
```bash
pip install -r requirements.txt
