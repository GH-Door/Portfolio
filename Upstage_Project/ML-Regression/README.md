# AI 부트캠프 13기 ML경진대회 3조

## 0. Overview
![이승민](./img/subtotal.png)


## 1. Competiton Info

### Timeline
- 2025.05.01 - 2025.05.15

## 2. Components

### Directory

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   └── pdf
│        └── (Template) [패스트캠퍼스] Upstage AI Lab 13기_그룹 스터디 .pptx
│   
└── data
    ├── eval
    └── train
```

## 3. Data descrption

### Dataset overview

- _Explain using data_

### EDA
- 자치구별 평균 아파트 거래가
![](./img/자치구_평균_아파트_거래가.png)
- Top30 아파트 브랜드별 평균 거래금액
![](./img/Top30_아파트_브랜드별_평균_거래금액.png)
- 브랜드 vs 일반 아파트 실거래가 분포
![](./img/브랜드_vs_일반_아파트_실거래가_분포.png)
- target_분포도_층_산점도
![](./img/target_분포도_층_산점도.png)
- 구별 아파트 가격 Boxplot 계약년월 변화
![이승민](./img/regen_target_time_animation.gif)
- 건물연령대별 및 주요 3개 시군구별 가격 분포
![](./img/건물연령대별_및_주요_3개_시군구별_가격_분포.png)
- 대출금리별 아파트 가격 & 거래 밀도
![이승민](./img/신용_대출금리.gif)
![이승민](./img/전세_자금대출금리.gif)
![이승민](./img/주택_담보대출금리.gif)
![이승민](./img/평균_대출금리.gif)
- 평균 대출금리 구별 아파트 가격 & 거래 밀도
![](./img/서울_주요_3개구_연도별_3x3.gif)
- spearman 상관계수
![](./img/spearman_상관계수.png)

### Feature engineering
- Mutual Information 중도가 낮은 변수들만을 교집합 으로 타겟 예측에 기여하지 않은 피처들 제거
  ![](./img/mutual_information.png)
- 범주형 최빈값 비모수 검정으로 범주형 변수들 유의성 확인
![](./img/범주형_최빈값_비모수_검정.png)

### Data Processing

- _Describe data processing process (e.g. Data Labeling, Data Cleaning..)_

## 4. Modeling
- 문국현
  - 검증셋 분할 방법 : TimeSeriesSplit 사용
  - 이유: Test Data엔 2023년 7~9월까지 데이터만 있다.
  ![](./img/modeling_mgh.png)
- 조선미
  ![](./img/modeling_jsm.png)
  - Feature importances : 전용면적, 건축년도, 계약년도, 가까운 대장 아파트 거리, 1km 이내 병원수, 가까운 백화점 거리, 가까운 지하철거리
  ![](./img/modeling_jsm_result.png)
- 홍정민
![](./img/modeling_hjm.png)
- 이승민
![](./img/modeling_lsm.png)
- 문진숙
![](./img/modeling_mjs.png)
  - SHAP
![](./img/modeling_mjs_result.png)

### Model descrition

- _Write model information and why your select this model_

### Modeling Process

- _Write model train and test process with capture_

## 5. Result

### Leader Board

![3조](./img/leaderboard.png)
- 4th, Public Score : 11593.6103