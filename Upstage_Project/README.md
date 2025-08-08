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
1. 데이터 점검/EDA: CSV-이미지 매칭 확인, 클래스 분포/불균형, 해상도·비율, 밝기·마스킹, 회전 상태 분석
2. 전처리/정규화: Test 분포에 맞춘 밝기/대비 보정, 자동 회전/왜곡 보정, 배경 정규화, 텍스트 가독성 향상, 클래스 가중/증강 설계
3. 모델링/학습: EfficientNetV2·ConvNeXt(V2) 실험, 이미지 크기 확장, 5-Fold·다중 시드, 캐싱 최적화, Hydra/W&B로 실험 관리
4. 추론: TTA 조합 설계 및 앙상블로 성능 안정화

### Lesson and Learned
- 문서 특유의 밝기/회전/마스킹 불일치가 성능에 미치는 영향 큼 → 도메인 맞춤 증강·정규화 전략이 핵심
- K-Fold와 TTA가 안정성과 재현성에 기여, 설정 관리로 협업/재현성 향상

- 회고록: [Upstage AI Lab CV 경진대회 회고](https://gh-door.github.io/posts/bootcamp-DL-16/)

<br>

## [NLP-Summarization: 일상 대화 요약](/Upstage_Project/NLP-Summarization)

### 문제 정의
- 일상 대화를 효과적으로 요약하는 생성 모델 개발. 화자 토큰과 개인정보 마스킹 등 특수 토큰 처리를 포함한 한국어 대화 요약.

### 주요 작업
1. 데이터/토크나이저: #PersonN#·PII 마스킹 토큰을 special tokens로 등록, BOS/EOS 및 길이 제약 설정, Train/Inference 입출력 전처리·후처리 구성
2. 모델링/학습: KoBART 파인튜닝, AdamW+Cosine, FP16, Early Stopping·주기적 평가, 최고 성능 체크포인트 관리
3. 생성/평가: Beam Search, no-repeat n-gram, 길이 제약으로 품질-속도 균형, ROUGE 기반 성능 모니터링(W&B)

### Lesson and Learned
- 화자/개인정보 토큰 보존이 요약 품질에 필수, 토크나이저 설정이 직접적 영향
- Beam 크기·길이 제한·n-gram 제약 등 생성 하이퍼파라미터의 품질-지연 트레이드오프 체감

- 회고록: [Upstage AI Lab NLP 경진대회 회고](https://gh-door.github.io/posts/bootcamp-DL-20/)