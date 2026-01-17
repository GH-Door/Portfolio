<div align="center">

<h1>외국인 대상 영문 법령 검색 AI</h1>

</div>

**💭 Language : Python**

**🛠 Tool : Google Colab**

**📅 진행기간 : 2024.06.17 ~ 2024.07.11**

**👥 인원 : 5명**

----------------------------------------------------------------

# 프로젝트 개요
- 법령 정보의 복잡성과 언어 장벽으로 인해 외국인들이 법령을 이해하는 데 어려움을 겪음
- 외국인 대상 범죄가 자주 발생 하지만 범죄 관련 기타 다른 법령 검색은 쉽지 않다.
- 그렇기에 외국인을 위한 맞춤형 영문 법령을 쉽게 찾아볼 수 있는 법령 추천 AI를 생성

<br><br>

# 프로세스
<p align="center">
 <img src="https://github.com/user-attachments/assets/bb40df48-50a2-4092-947f-7379509077ea" width="1000">	
<p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/849ba289-4f06-4ed8-9656-a4a31b16ad79" width="800">	
<p>
	
<br><br>


# 주요 역할
- 데이터 수집 및 전처리
- TF-IDF 방식으로 데이터 벡터화
- 코사인 유사도를 통한 텍스트 간 유사도 계산 및 상위 유사 항목 도출
- 상위 유사 텍스트 중 가장 자주 등장하는 법령명 분석
- 법령 최적화를 위해 다양한 n-gram 과 텍스트 전처리 기법 활용

<br><br>

# 데이터 수집 및 전처리
<p align="center">
 <img src="https://github.com/user-attachments/assets/e45f06e5-0693-469b-9629-b222a63e00e5" width="1000">	
<p>

<br>

- 국가법령정보 공동활용의 영문법령 **API**를 활용하여 추출 : **[링크](https://law.go.kr/engLsSc.do?menuId=1&subMenuId=21&tabMenuId=117&query=)**

- 법령 본문과 기본정보로 분리해서 추출 / 총 **2691**개의 법령, **12만개**의 조항 확인


- 법령 중 외국인 비대상 법령 제거  
	 ↳  Ex:) **군사, 평창, 공무원, 전쟁, 운영, 국가 유공자, 공직자** … 등의 키워드를 포함한 법령

- 조항 내 의미 없는 키워드 제거(노이즈 제거)  
	 ↳  Ex:) **Definitions, Purpose, decree, presidential** 등 같은 단어가 반복적으로 등장, 높은 빈도로 나타남(중요한 키워드의 가중치를 낮추고 전체 모델의 성능 저하 우려)

- 불용어 제거  
	 ↳  Ex:) **and, is, it, by, the 등 과 같은 불용어** 제거,  모델의 성능을 향상시키고, 처리 속도를 증가
 
<br><br>


# ML 모델 선택 및 학습
## 1. TF IDF 유사도 모델
<p align="center">
 <img src="https://github.com/user-attachments/assets/b3162538-31d1-48a7-9af7-c1370387cc6c" width="600">
<p>

<p align="center">
 <img src="https://github.com/user-attachments/assets/9927aa69-ae21-4f57-bdd2-0360d2fa3131" width="800">
<p>

<br>

### 모델 선택 이유

- ↳ 핵심 단어 추출 : **TF-IDF**는 각 단어의 문서 내 중요도를 평가하여 문서 간 차이가 존재 
  TF-IDF를 사용시 각 문서를 단어의 **벡터**로 표현하여 문서 주요 특징 추출 가능
- 코사인 유사도 계산 : **TF-IDF** 벡터를 사용하여 **문서 쌍 간의 유사도 계산**

- 과정 : Cleaning 벡터화 과정 중 필요없는 단어 2차 정제, **Tokenization**은 **nltk**의 **tokenizer** 사용 


- 문장 입력 : **"My passport was stolen, and I need help reporting it and obtaining a new one”**
	            "제 여권을 도난 당했습니다. 신고하고 새 여권을 발급받는 데 도움이 필요합니다.”

   		↳ TOP1 : 여권법 시행령	- 내용 : 여권 기재 정보 및 기재 방법
  		  TOP2 : 여권법		- 내용 : 여권 서비스 및 반환에 대한 조항
  		  TOP3 : 여권법		- 내용 : 유효한 여권 발급 및 요효기간에 관한 조항

<br><br>

# DL 모델 선택 및 학습(BERT 기반 문장 유사도 모델)
<p align="center">
 <img src="https://github.com/user-attachments/assets/472dd12f-898f-40c1-991b-b03657f8b07d" width="800">
<p>

## 모델 선택 이유

- 문맥적 이해 : BERT 는 문장의 전후 맥락을 모두 고려하여 임베딩을 생성하기에, 문장의 의미를 깊이 있게 이해 가능
- 유연성 : 문장 분류, 문장 유사도, 질의응답 등 다양한 태스크에 쉽게 적용할 수 있는 유연한 구조


<br>

## 2-1. Keyword 기반 Faiss 유사도 모델
<p align="center">
<img src="https://github.com/user-attachments/assets/7662a342-75ef-4c8a-8d18-365ecad141c0" width="800">
<p>

- 빠른 검색을 위한 **Faiss 유클리안(거리기반)** 검색 / Embedding Model은 **bert**기반 기존 **Pre-trained model** 사용


<pre>
↳ TOP1 : 여권법          - 내용 : 일회용 여권 관련 조항
  TOP2 : 여권법 시행령   - 내용 : 긴급한 인도적 사유로 인한 예외적 여권 발급
  TOP3 : 여권법          - 내용 : 처벌 관련 조항
</pre>

<br>

## 2-2. Keyword 기반 Embedder Fine tuning
<p align="center">
<img src="https://github.com/user-attachments/assets/be92675b-80a0-465d-84a3-dac41c3b3924" width="800">
<p>

- 공통 키워드 존재 여부로 0 / 1 라벨링 > Text-classification 모델 사용하여 긍 부정 학습


		↳ TOP1 : 간접투자자산운용법 시행령 	- 내용 : 투자 자산운용업법 시행 목적
  		  TOP2 : 간접투자자산운용법 시행규칙	- 내용 : 기준가격 차이의 허용 범위
  		  TOP3 : 간접투자자산운용법 시행규칙	- 내용 : 종합 금융회사에 적용되는 특별 규칙
   
<br>

## 2-3. Keyword Cluster 및 Embedder Fine tuning
<p align="center">
<img src="https://github.com/user-attachments/assets/63d52a45-9efb-493c-83ee-3a54843922cc" width="800">
<p>

- Keyword Embedding 후 UMAP을 통해 차원축소 > HDBSCAN Clustering 진행, 총 87개의 클러스터 확인
- 각 조항의 클러스터 공통 여부에 따라 0 / 1 로 라벨링  > Embedding 후 Cosine 유사도와 라벨 비교를 통해 학습 (MSE Loss)


		↳ TOP1 : 출입국 관리법 	- 내용 : 통지 처분 방법
  		  TOP2 : 여권법 시행령	- 내용 : 요금 관련 조항
  		  TOP3 : 출입국 관리법	- 내용 : 이민 위반자 신고 관련 조항
   
<br><br>


# 최종 모델(앙상블)
<p align="center">
<img src="https://github.com/user-attachments/assets/f9e4594f-aa99-4022-aff8-085dcfeaa910" width="800">
<p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/59ce11bb-6b1f-43d2-b2aa-ef8b251666db" width="800">
<p>


- 위 4개의 모델의 결과를 바탕으로 최빈 법령 탐색
- 최빈 법령에 포함된 법령으로 검색 범위 축소 > 최빈 조항 순으로 출력

<br><br>

# 모델 평가
<p align="center">
 <img width="1100" alt="image" src="https://github.com/user-attachments/assets/ce4ed33a-df75-45c2-8770-a56fdc00ac6d">
<p>
	
### 평가 지표

- 유사도 측정: 생성된 문장 임베딩을 바탕으로, 실제 법령 데이터와 비교하여 모델이 얼마나 유사 문서를 식별하는지 측정
- 성능 평가: 모델의 유사도 측정 결과를 바탕으로, 정확도, 정밀도 및 재현율과 같은 통계적 지표를 사용하여 모델 성능을 평가
- Keyword Cluster 추출 및 Embedder Fine tuning 모델은 비지도 학습의 특성상 일반적인 성능 지표보다는 의미적 유사성과 군집 품질에 중점을 두어 평가

<br><br>

# 결과

- 검색한 문장과의 유사도 검사를 통해 관련 법령 검색 (4가지 모델의 결과를 Voting 후 최종 결과 도출)
- 각 모델의 결과보다 최종 앙상블 모델의 결과가 더 다양하게 나타난다.
- 각 모델 결과는 단어 자체에 집중하여 비슷한 분야라 하더라도 물어본 질의와는 다른 결과를 나타낼 때가 많았다.
- 반면 최종 모델의 경우 입력해본 몇몇 문장에 한하여 물어본 질의에 근접한 조항을 찾는다.

<br><br>

# 기대효과 및 Lesson and Learned

## 기대효과
- 외국인을 위한 맞춤형 법령 검색 시스템 구축을 통해, 외국인들이 법령을 쉽게 이해하고 관련 법령을 신속하게 찾을 수 있음
- 외국인 관련 법적 분쟁 및 범죄 발생을 줄이는 데 기여
- ML 및 DL 모델을 결합한 AI 시스템을 통해 법률 전문가들이 법령 검토 시간을 단축할 수 있음

## Lesson & Learned
- 데이터 전처리와 텍스트 처리의 중요성을 이해하고, 다양한 전처리 기법을 학습
- TF-IDF 및 코사인 유사도를 통한 텍스트 유사도 분석 과정에서 텍스트 데이터의 특성을 다루는 방법을 터득
- 머신러닝 및 딥러닝 모델을 결합하여 최적의 성능을 내기 위한 모델 선택과 튜닝의 중요성을 학습
