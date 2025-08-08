## [ML-Regression: 아파트 실거래가 예측](/Upstage_Project/ML-Regression)

### 문제 정의
- 정량 지표(연속형 타겟)를 예측하는 회귀 문제. 서울시 아파트 실거래가를 예측하는 모델을 개발하여 일반화 성능을 극대화.

### 주요 작업
1. 데이터 수집/전처리: 공공데이터 수집, 주소를 위경도로 변환(카카오 API), 시설물과 단지 간 거리 계산, 결측치 RandomForest 보간
2. EDA: 수치형 정규성 점검, 이상치 탐지, 범주형 변수별 평균 거래가 분석
3. 피처 엔지니어링: 인프라 밀집도/시간 파생변수, Spearman/MIScore 분석, Kruskal–Wallis로 유의 변수 선별, Label+Frequency 인코딩
4. 모델링/튜닝: LightGBM·XGBoost·CatBoost 등 트리 계열 비교, K-Fold 교차검증, 베이지안 최적화
5. 최종 전략: 상위 모델 TOP3 소프트 보팅 앙상블

### Lesson and Learned
- 시계열 단절을 고려한 Time-based split 검증의 중요성 확인, 임의 Hold-Out 대비 제출 점수 일치도 향상
- 검증 설계가 하이퍼파라미터/피처 선택 전반에 미치는 영향 체감, 데이터 누수 방지 원칙 강화
- 대용량 산출물은 버전관리에서 배제하고 재현 가능한 파이프라인 유지
- 회고록: [Upstage AI Lab ML 경진대회 회고](https://gh-door.github.io/posts/ML-contest/)

<br>

## [CV-Classification: 문서 타입 이미지 분류](/Upstage_Project/CV-Classification)

### 문제 정의
- 문서 이미지를 17개 클래스 중 하나로 분류하는 모델 개발. 다양한 회전/왜곡/밝기 분포를 갖는 실제 문서 환경을 가정하여 높은 정확도와 안정성을 확보.

### 주요 작업
1. 데이터 점검/EDA: CSV-이미지 매칭 확인, 클래스 분포/불균형, 해상도·비율 확인
2. 증강 설계: 회전·밝기·크롭 등 조합으로 10배 증강 구성, 검증 절차에 반영
3. 모델링/학습: ConvNeXtV2-Base 중심 실험, 이미지 크기 확장, 3-Fold·다중 시드, W&B로 실험 관리
4. 추론: TTA 조합 설계 및 앙상블로 성능 안정화

### Lesson and Learned
- 증강과 TTA 설계가 문서 분류 성능과 안정성에 핵심적으로 기여
- Fold 설계(3-Fold)와 다중 시드가 점수 변동성 완화에 효과적
- 실험 로깅과 설정 고정(W&B)이 재현성과 협업 효율을 높임
- 회고록: [Upstage AI Lab CV 경진대회 회고](https://gh-door.github.io/posts/bootcamp-DL-16/)

<br>

## [NLP-Summarization: 일상 대화 요약](/Upstage_Project/NLP-Summarization)

### 문제 정의
- 일상 대화를 효과적으로 요약하는 생성 모델 개발. 화자 토큰과 개인정보 마스킹 등 특수 토큰 처리를 포함한 한국어 대화 요약.

### 주요 작업
1. EDA: 대화/요약 길이 분포, 주제(topic) 분포, 화자 수 분포 파악
2. 데이터 전처리: 텍스트 정규화, 토크나이저 설정, 시퀀스 길이·패딩/절단 전략 수립
3. 모델링/학습: KoBART 파인튜닝(AdamW + Cosine, FP16), Early Stopping 적용, ROUGE로 성능 평가
4. 실험 관리: W&B로 실험 로깅 및 비교, 체크포인트 관리

### Lesson and Learned
- 토크나이저와 시퀀스 길이 설정이 생성 품질에 직접적인 영향
- 학습 안정화(Early Stopping, 스케줄러, FP16)가 성능 재현성과 개발 속도에 기여
- 모델/토크나이저 버전 고정과 실험 관리가 환경 차이에 의한 오류를 예방
- 회고록: [Upstage AI Lab NLP 경진대회 회고](https://gh-door.github.io/posts/bootcamp-DL-20/)