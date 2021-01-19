#  A Survey on Application of Knowledge Graph
### Published
- CCEAI 2020
- Journal of Physics
### Author
- Xiaohan Zou
- 중국 상하이 Tongji Univ. 소프트웨어 공학 전공
---
## Abstract
- 지식그래프란?
  - 정보를 semantic graph로 표현한 것 (representation)
- 활용분야
  - Question Answering
  - Recommendation
  - Information Retrieval
  - commercial and scientific domains
- 본 논문은 각기 다른 도메인에서의 application에 대한 첫 survey.

## 1. Introduction

### 지식그래프의 발전 역사
- KG -> 지식 구조화가 필요한 Information System에 많이 사용되어 옴 (base가 됨)
- semantic web 등장 (2001)
- 초기에는 RDF 적용하여 그래프 기반 지식 표현
  - *RDF : Resource Description Framework*
  - Node는 entity
  - Node들은 relation을 의미하는 edge로 연결
  - Relation set은 스키마나 온톨로지로 표현 가능
- Linked Data 등장 (2009)
  - 시멘틱웹의 각기 다른 데이터셋 연결
- Knowledge Graph 등장 (2012, 구글)
  - 사용 목적
    1. 텍스트 내 entity를 구분 (identify, disambiguate)
    2. 시멘틱하게 구조화한 summary로 검색 결과를 enrich
    3. 관련된 entity에 링크 제공하여 검색엔진으로 활용
  - 이후 마이크로소프트의 검색엔진 Bing에서도 비슷하게 KG 이용
  - 요즘은 DBpedia, YAGO, Wikidata, Freebase 같은 semantic web knowledge bases도 KG를 참조

### 본 논문의 핵심
- KG은 산업적, 학문적으로 크게 영향을 미쳤으나, 지금까지는 KG를 구성하는 **construction techniques**에만 초점을 둔 review가 많았음
  - 추출, 표현, 퓨전(융합), 추론에 초점 (extraction, representation, fusion, reasoning)
- 따라서 본 논문은 KG의 다양한 application에 초점
---
## 2. Application
## 2.1 Question Answering System
- semantic-aware QA 서비스에 쓰일 수.
- 예시
  - Watson : YAGO나 DBpedia를 데이터로 사용
  - 챗봇, 디지털 비서 (XiaoIce, Cortana, Siri)
### 1. Semantic Parsing 기반 QA
  - 방식
    - 자연어 질의 -> logic 형태로 변환 -> 구조화된 쿼리 생성 (예) SPARQL
      - 이 logic 형태는 전체 쿼리의 시멘틱으로 표현 가능
      - SPARQL
        - DB를 위한 시멘틱 질의어
        - RDF 형식의 데이터 검색, 조작 가능
  - 연구예시
    - 한 연구에서는 구절(phrase)와 술어(predicates)를 맵핑
  - 장단점
    - 장점 : 복잡한 질문에 성능 좋음
    - 단점 : semantic parser 위해 hand-crafted features 필요하기에 도메인과 확장성에서 한계가 있음
      - *hand-crafted features : 사람이 직접 설계한 특징*
### 2. Information Retrieval 기반 QA
  - 방식
    - Question -> 자동으로 구조화된 쿼리로 바꿈
    - Answer -> 지식베이스에서 candidate answer set 추출
    - Question과 Answer의 피처들을 각각 추출하여 answer candidate들에 랭킹 매김
  - 연구예시
    - question은 question feature graph로 변환, candidate answer는 지식베이스에서 구성된 topic graph 내 각 노드
  - 장단점
    - 장점 : 자연어 질의의 시멘틱을 크게 고려하지 않아도 됨. 간단 쿼리에서는 좋은 성능
    - 단점 : Question에 대한 피처 추출하기 위해 rule과 dependency parse 결과에 의존함
### 3. Embedding 기반 QA
  - 연구예시
     - (1) Question -> low-dim 벡터 임베딩 학습 후, 지식베이스에서 question과 candidate answer 간 유사도 계산. 가장 높은 유사도 점수의 candidate answer가 최종 answer.
     - (2) embedding 기반 모델을 fine-tune하여 성능 개선
       - 임베딩 space 내 유사도를 matrix parameterizing하는 부분에서 최적화
   - 장단점
     - 장점 : 위 방법들에 비해 hand-crafted features, POS 태깅 위한 추가적인 system, 구문분석 등을 학습에 사용하지 않고도 좋은 성능
     - 단점 : 단어 순서를 고려하지 않음, 복잡한 질문 처리 불가.
### 4. 딥러닝 기반 QA
  - 연구예시
    - (1) Multi-Column CNN : candidate answer에 대해 랭크를 매기는 score layer 추가
    - (2) End-to-End NN with cross-attention mechanism
    - (3) Query graph generation을 staged search problem으로. CNN도 적용
    - (4) Attention 기반 BiLSTM
### 5. 더 복잡한 QA
  - (1) 다른 연구들은 fact-finding extractive QA 였다면, 또다른 연구는 multi-hop generative task에 초점
    - multi-attention mechanism을 이용해 추론에 multiple hops 수행
    - answer -> pointer-generator decoder로 합성
  - (2) Code-Mix Simple Questions QA
    - TSHCNN 이용하여 candidate answers의 랭킹을 다시 매김 (re-rank)
    - K-Nearest and Bilingual Embedding 을 이용하여 language transformation
  ---
## 2.2 Recommender System
- 전통적인 추천 방법 : Collaborative Filtering (CF)
  - 유저의 취향, 과거 interactions 기반으로 함
  - 단점
    - 유저 데이터의 sparsity(interaction 정보가 부족할 때)
    - cold start problem (유저나 아이템에 대한 정보가 없을 때)
- 위 단점 개선 방법이 **side information**을 이용하는 것 (부가정보)
- 이런 부가정보를 활용에 KG를 적용하는 연구가 활발
  - 특히 KG의 relation을 통해 추천 정확도를 높이고 아이템 다양성을 확장시킬 수.
  - 또한 해석력을 부여함. (interpretability)
### 1. Embedding Based 추천
- 방식
  - (1) **Preprocessing** : Knowledge Graph Embedding (KGE)  알고리즘을 이용해 KG를 Preprocessing
  - (2) **Apply Entity Embedding** : 학습한 Entity 임베딩을 추천 프레임워크에 적용
- 연구예시
  - (1) DKN : CNN 기반으로, Entity 임베딩과 Word 임베딩을 결합하여 뉴스 추천
  - (2) Unified Bayesian Framework : CF 모듈과 아이템의 Text 임베딩, image 임베딩, knowledge 임베딩을 결합
  - (3) Deep Autoencoder를 이용하여 social, profile, sentiment 네트워크 develop
  - (4) cross & compress unit 디자인하여
    - KGE와 추천 태스크의 latent feature를 자동으로 공유
    - 추천시스템의 아이템과 KG의 entity간의 높은 상호작용 학습(high-order interaction)
- 장단점
  - 장점 : 추천시스템 적용에서의 high flexability
  - 단점 : text를 제외하고는 side info를 가지기 어려움, 추천하기에 충분하게 적합하지는 않음
### 2. Path Based 추천
- direct 하게 그래프 알고리즘 만들어 node 간 다양한 연결 패턴을 통해 추천에 부가 정보 (additional info) 제공
- 연구예시
  - (1) KG는 heterogeneous info-based network로 여겨짐
    - *heterogeneous : 여러 다른 종류들로 이루어진*
    - KG로부터 meta-graph / meta-path 기반 latent features를 추출하여 서로 다른 종류의 relation graph / path를 연결하는 아이템과 유저 간의 링크를 표현
  - (2) LSTM 이용하여 각 path에 대해 representation 생성
    - entity와 relation의 시멘틱 구성
    - path 내의 순차적 종속성을 활용하여 path에 대해 효과적인 추론 가능하게 함
  - 장단점
    - 장점 : 자연스럽고 직관적인 KG 활용
    - 단점 : handcrafted designed mata-path에 의존이 심함
      - 이런 meta-path는 실제로는 최적화 하기 어렵고 특수 시나리오에서는 디자인이 어려움
        - 예) entity와 relation이 동일 도메인 내에 있지 않은 '뉴스 추천'
### 3. 기타 추천 관련 연구
- (1) RippleNet : embedding 기반 연구와 path 기반 연구의 장점을 합치려고 함
  - *ripple : 잔물결, 파문. 퍼져나가는 것이라고 이해하면 될 듯*
  - KG에서 유저의 잠재적인 선호도를 propagate (전파)하여 hierarchical interest를 발견
  - 여기서 선호 정보를 propagate 한다는 점이 KGE 방식을 통합했다고 볼 수 있으며, hand-crafted design 필요하지 않음.
  - 단점 : relation을 덜 고려함. KG의 사이즈가 커질수록 ripple set의 크기가 지나치게 커져서 계산량이 크고 저장공간에 오버헤드 발생.
- (2) 개선된 유저-아이템 모델링을 통해 결측 fact들 보완 (?)
  - 이를 보조 데이터로 사용하여 유저-아이템 interaction의 모델링을 augment하는 데 활용 (?)
---
## 2.3 Information Retrieval
- 웹 기반 검색엔진에 KG로부터 entity data를 활용하여 검색 성능을 높이고 있음.
- 검색엔진에서의 적용 예시
  - 구글 -> Google Plus, Google Knowledge Graph
  - 페이스북 -> Graph Search 활용
- KG를 쿼리와 문서를 이해하는 데 활용할 수 있다.
- KG의 시멘틱을 다양한 요소에 적용 가능
  - (1) Query Representation
    - 관련 entity와 text를 query 확장에 반영
    - 연구예시
      - entity를 지식 베이스에 link -> 구조화된 attributes와 text를 이용해 query enrich
  - (2) Document Representation
    - 문서의 벡터 공간 모델 (vector space model)에 annotated entities를 추가 (주석처리된 엔티티 (?))
    - 연구예시
      - query와 document의 entity annotation 이용 -> bag-of-entities 벡터 생성,
      - entity space에서의 문서와 쿼리 간의 output match를 이용해 문서 랭킹에 활용.
  - (3) 검색 시스템에서의 Ranking
    -  query에서 document로 additional connection 생성하여 랭킹모델 개선
    -  연구예시
       -  query와 document를 고차원으로 맵핑 후 투영
       -  이 때 각 차원은 하나의 entity
       -  이후 각 차원으로 투영한 (projection) latent space에서의 query와 document 간의 관련도 측정
  - (4) 딥러닝 이용 Information Retrieval
    - large-scale train data에서 더 복잡한 랭킹 모델 학습 가능
    - 연구예시
      - Neural Search System에서의 KG
        - KG로부터 시멘틱 통합, interaction-based neural ranking networks 이용해 문서 랭킹
---
## 2.4 도메인 별
1. 의료 분야
2. 사이버 보안
3. 금융
4. 뉴스
- 뉴스 : 다이나믹, 시간에 따라 변화하고, 정보가 압축적이라는 특징이 있음
- 이러한 특징을 다루고자 KG 활용
- 연구예시
	- (1) DKN : 뉴스에서의 latent knowledge-level connection을 찾기 위해 KG 활용
	- (2) 이벤트 중심 KG (event-centric)로 세계의 변화 관련 뉴스를 다양한 언어로 표현
	- (3) 비구조화된 '뉴스 기사'와 구조화된 '위키데이터'를 결합하여 이벤트를 표현하는 뉴스 기사 추출
	- (4) 가짜뉴스 탐지 : KG에서의 link-prediction task로 다룸
		- Factual Statements network에서 이질적인 연결 패턴 찾아냄
- 저자의 견해 : 뉴스는 빠르게 서로 다른 나라로 전파되기에 **entity resolution**, **semantic role labelling**이 특히나 multilingual 환경에서 중요
5. 교육
- 학습자료 추천, 컨셉 시각화에 KG 활용
- 연구예시
	- (1) KnowEDU : 교육을 위한 KG 자동으로 구성
		- node -> 학습해야하는 교수학적 컨셉, 이런 컨셉 추출하기 위해 RNN 모델 적용
		- relation -> 학생들의 performance 데이터에서의 연관규칙분석 이용하여 식별됨
	- (2) 학습 자료 추천, 통합하는 툴에서 KG를 사용하여 시멘틱 표현, 중요 컨셉 식별
- 저자의 견해 : 단순히 relation 추출하는 것에 초점을 두지 말고 깊이 있고 정확하게 추출하여 latent info 발견에 도움이 될 것
6. 기타 applications
