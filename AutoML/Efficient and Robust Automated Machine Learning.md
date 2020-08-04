# Efficient and Robust Automated Machine Learning

## Abstract

- AutoML
	- automated machine learning
	- 새로운 데이터셋에 대해 좋은 알고리즘, 피처 전처리 단계, 각 각의 하이퍼파라미터 값을 선택하는 것
	- Bayesian Optimization 방법론 이용
- auto-sklearn
	- scikit-learn 기반
	- 15개의 classifiers, 14개의 feature preprocessing methods, 4개의 data preprocessing methods를 사용하여 110개의 하이퍼파라미터들로 구성된 structured hypothesis space

## 1. Introduction

- 머신러닝이 다양한 분야에 활용됨에 따라 쉽게 이용할 수 있는 머신러닝 시스템에 대한 수요가 높아지고 있다
- 이 때, 모든 머신러닝 서비스는 가장 **근본적 문제**를 해결해야 한다
	- 주어진 데이터셋에 대해 어떤 머신러닝 알고리즘을 사용할 것인지
	- 피처는 어떻게 전처리 할 것인지
	- 하이퍼 파라미터는 어떻게 설정할 것인지
- 논문에서는 AutoML을 다음과 같이 정의한다 :
	- automatically producing test set predictions for a new dataset within **fixed computational budget**
	- 조금 더 수식적으로 적으면 다음과 같다
		- feature vector $x_i$, 대응되는 target value $y_i$
		- train data $D_{train} = {(x_1, y_1), ... , (x_n, y_n)}$
		- train data $D_{test} = {(x_{n+1}, y_{n+1}), ... , (x_{n+m}, y_{n+m})}$
		- these are drawn from the same underlying data distribution
			- (train & test 데이터는 동일한 모집단에서 추출되었다고 가정한다는 것)
		- 즉, AutoML 문제는 test 데이터의 예측값인 $\hat{y}_{n+j}, ..., \hat{y}_{n+m}$을 자동으로 구하는 것
			- 이 때 Loss 는 $\frac{1}{m} \sum_{j=1}^m L(\hat{y}_{n+j}, y_{n+j})$
			- m개의 각 테스트 데이터 Loss의 전체 평균
- 위에서 **fixed computational budget** 표현이 있었는데, butget ($b$)란?
	- budget은 computational resources (연산 자원 의미)
	- CPU, 시간, 메모리 사용량
	- 본 auto-sklearn이 이 budget 문제를 반영한 ChaLearn AutoML 대회를 우승했다고 함
- Auto-WEKA에서 motivated
	- Auto-WEKA는 머신러닝 프레임워크인 WEKA와 Bayesian optimization method (BO)를 결합하여 주어진 데이터셋에 대해 좋은 WEKA instantiation을 선택
- 본 논문의 목적은 efficiency와 robustness를 향상시키기 위함
	- 광범위한 머신러닝 프레임워크에 적용되는 기본 principles에 기반
- 논문 순서
	- 1. 새로운 데이터셋에 대해 잘 perform하는 머신러닝 프레임워크 instantiation 를 찾기위해 데이터셋들을 reason한 뒤, 이를 이용해 BO warmstart
	- 2. BO 결과를 고려하여 모델 앙상블을 자동 구성
	- 3. scikit-learn을 이용하여 highly parameterized 머신러닝 프레임워크 디자인
	- 4. auto-sklearn이 다른 AutoML SOTA보다 좋다 증명
	
## 2. AutoML = CASH

- 본 논문은 AutoML을 CASH라고 표현
	- **C**ombined **A**lgorithm **S**election and **H**yperparameter optimization의 약자
	- 알고리즘 선택 & 하이퍼파미터 최적화를 함께 한다는 의미인 것 같다
- AutoML에서 중요한 것이 두 가지 있다고 말하는데,
	- 첫째, 모든 데이터셋에 대해 잘 작동하는 하나의 머신러닝 기법은 없다는 것
	- 둘째, 머신러닝 기법의 일부는 하이퍼파라미터 최적화에 영향을 많이 받는다는 것 (crucially rely on HO)
	- 여기서 이 두 번째 문제는 BO를 통해 해결되고 있어서, BO는 AutoML에서 중요
	- 이 두 가지는 서로 얽혀있다
		- 이유? 알고리즘 랭킹은 하이퍼파라미터가 잘 튜닝되었는지에 따라 영향을 많이 받기 때문
	- 하지만 이 두 문제는 하나의 조인트 최적화 문제로 다루어질 수 있다고 한다.
		(single, structured, joint optimization problem)
- 여기서 또 CASH 문제를 수식적으로 정의하면, 다음과 같다.
$$argmin \frac{1}{K} \sum_{i=1}^K L(A^{(j)}_\lambda,  D^{(i)} _{train}, D^{(i)}_{valid})$$
- 즉, train 데이터를 이용해 알고리즘 A와 파라미터 $\lambda$를 학습하여 valid 데이터에 대해 예측한 K-fold validatation의 Loss 평균을 최소화 하는 것이 CASH 이자 논문에서 말하는 AutoML인 것이다.
- Bayesian Optimization
	- 정의
		- (1) 하이퍼파라미터 셋팅과 성능 간의 관계를 찾기위해 확률 모델을 학습
		- (fits a probabilistic model to capture the relationship between hyperparameter settings and their measured performance)
		- (2) 이 모델을 이용해 가장 좋아보이는(most promising) 하이퍼파라미터 셋팅을 선택해 평가하고, 모델을 업데이트 하는 과정을 반복한다.
		- 이 때, 논문은 **공간의 새로운 부분에 대한 탐색** & **알고 있는 공간에 대한 사용** 간의 trade-off라고 표현하는데 무슨 표현인지 잘 모르겠다.
	- 종류
		- (1) 가우시안 프로세스 기반 BO 모델
			- 수치적 하이퍼파라미터의 low-dimensional 문제에서 성능이 좋다
		- (2) 트리 기반 BO 모델
			- high-dimensional, structured, 그리고 partly discrete 문제에서 성능이 좋다
			- e.g. CASH 모델
			- 트리 기반 BO 모델 중에서도 SMAC은
				-  랜덤포레스트 기반의 모델로, 성능이 좋다고 알려짐
				- 또한 SMAC의 주요 특징은 cross-validation을 빠르게 수행
					- 이유:한번에 한 fold를 평가하고, 성능이 낮은 하이퍼파라미터 세팅은 진작에 버림
- 따라서 본 논문은 SMAC 모델을 사용하여 CASH문제 해결

## 3. New methods in this papaer
- New methods for increasing efficiency and robustness of AutoML
- 논문에서는 성능을 향상시킬수 있는 방법 두 가지를 제시한다.
	- 첫째, meta-learning 단계를 AutoML 파이프라인에 추가하여 BO 를 warmstart 할 수 있도록한다.
	- 둘째, 자동화된 앙상블 단계를 추가하여 BO로 찾은 모든 classifier들을 사용할 수 있도록 한다
- 이러한 방법은 자유도가 많은 ML 프레임워크에서 더 효과적일 것이라고 말한다.
	- 자유도가 많은 것의 예 ) 알고리즘, 하이퍼파라미터, 전처리 기법이 많은 것
	
### (1) Meta-learning

- 전문가들이 이전 태스크를 해결하며 경험을 쌓듯이, meta-learning도 마찬가지
- 본 논문에서는 meta-learning을 통해 새로운 데이터셋에 대해 성능이 좋은 머신러닝 프레임워크를 선택할 수 있도록 사용
- 더 자세히 말하면, 우선 많은 데이터셋들에 대해 (1) performance data (성능 결과 데이터), (2) meta-feature set를 가지고 있다.
	- 여기서 meta-feature라고 하는 것은, 새로운 데이터셋에 대해 어떤 알고리즘을 사용할지 참고할 수 있는 효율적으로 계산되는 데이터셋의 특징들
		- 논문에서는 38개의 meta-features 이용
		- simple, information-theoretic, statistical meta-features
		- e.g. 데이터 개수, 클래스 개수, skewness, entropy of the targets
- meta-learning은 BO와 상보적인 관계임을 이용
	- meta-learning
		- 장점) 빠르게 성능 좋은 ML 프레임워크 객체를 제안할 수 있다
		- 단점) 성능에 대해 면밀하게 정보 얻을 수 없다 (fine-grained info on performance)
	- BO
		- 장점) 시간이 지나며 성능을 fine-tuning 할 수 있다
		- 단점) 전체 ML 프레임워크 하이퍼파라미터 공간에 대해 진행하기엔 시작이 느리다
	- 따라서 meta-learning으로 얻은 k개의 configuration 결과를 사용해 BO 진행
---
- meta-learning 방법에 대해 조금 더 구체적으로 살펴보면 다음과 같다
	- (1) OpenML의 140개 머신러닝 데이터셋에 대해 meta-feature들을 미리 계산
	- (2) BO를 사용해 각 데이터셋에 대해 성능이 좋은 ML 프레임워크 객체를 저장해둔다
	- 본 논문에서는 BO에서 SMAC을 사용해 데이터셋 중 2/3로 24시간 동안 10-fold validation을 진행했고, 나머지 1/3 데이터로 성능 평가했을 때 가장 좋았던 ML 프레임워크를 저장
	- (3) 새로운 데이터셋에 대해
		- a. meta-feature들 계산
		- b. 계산된 meta-feature 공간에서 기존 데이터셋들과 새 데이터셋 간의 거리(=유사도) L1 계산
		- c. 상위 25개의 데이터셋들의 ML 프레임워크 객체를 선택해 평가 (k=25 nearest datasets)
		- d. 평가 결과에 대해 BO 진행

### (2) 모델 앙상블 자동화
- Automated ensemble construction of models evaluated during optimization
- Bayesian Hyperparameter Optimization
	- 장점) 좋은 성능의 하이퍼파라미터를 찾는 데에 data-efficient
	- 단점) 단지 예측을 잘하기 위함이라면 비효율적인 절차
		- 이유? 모델들이 많이 버려지니까 (좋은 성능의 모델 포함, 사용되지 않으니까)
- 본 논문에서는 모델을 저장하여 효율적인 post-processing method로 사용하는 방법을 제안, 두 번째 프로세스에서 즉시 실행 가능 (can be run in a second process on-the-fly)
- 자동 앙상블을 통해 단일 하이퍼파라미터 셋팅을 하지 않게 됨으로써 더욱 robust, 일반화(오버피팅 방지)
- 앙상블이 효과적인 경우
	- (1) 모델이 individually strong 한 경우
	- (2) 모델 간 에러가 uncorrelated 한 경우
- But, 단순히 BO로 찾은 모델들을 동일한 가중치로 앙상블 하는 경우, 잘 동작하지 않는다.
- 따라서, 각 모델을 hold-out 데이터셋으로 평가하여 가중치를 수정
- 가중치 최적화 방법 중 ensemble selection이 가장 fast, robust 했다고 함
	- ensemble selection
		- 빈 앙상블로 시작하여 모델을 하나씩 추가하는 greedy procedure
		- 앙상블의 validation 성능을 maximize 하는 방향으로 모델을 하나씩 반복적으로 추가
		- (가중치는 uniform하게 시작하지만 반복하며 달라지는 듯)
- 논문은 이 방법을 이용해 50개의 best 모델로 앙상블

## 4. A practival automated ML system

### Components
- 15개의 분류 알고리즘
- 14개의 피처 전처리 기법
- 4개의 데이터전처리 기법
- 각 각을 파라미터화 하여 총합 110개의 하이퍼파라미터 공간 만들어냄
- 대부분 조건부 하이퍼파라미터(conditional hyperparameters)이기에 각 요소들이 선택되어야만 active해지며, 이는 SMAC이 알아서 해줌

#### 1. 분류 알고리즘
- 15가지의 알고리즘은 아래 카테고리로 나누어짐
	- linear models
	- SVM
	- discriminant analysis
	- nearest neighbors
	- naive bayes
	- decision trees
	- ensembles
- base classifier로 공간 구성, 하나 이상의 base classifier로 파라미터화 된 meta-model이나 앙상블은 제외

#### 2. 전처리 기법
- 데이터 전처리 기법 (4개)
	- 데이터 전처리하면 피처 값들을 바뀐다
	- 언제나 적용됨
		- 인풋값 rescaling
		- 결측값 대체(imputation of missing values)
		- one-hot encoding
		- 타켓 클래스 balance 맞추기
- 피처 전처리 기법 (14개)
	- 실제 피처 set이 바뀐다
	- 아예 안 쓰일 수 있고, 쓰여도 오로지 하나만
		- feature selection
		- kernel approximmation
		- matrix decomposition
		- embeddings
		- feature clustering
		- polynomial feature expansion
		- sparse representation transformation 등
---
이후 논문의 내용은 다른 AutoML과 비교, 자랑하는 부분이라 생략.
