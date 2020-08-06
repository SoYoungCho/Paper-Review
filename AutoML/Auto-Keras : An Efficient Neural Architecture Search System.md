# Auto-Keras : An Efficient Neural Architecture Search System

## Abstract
- 현재까지 발표된 NAS 알고리즘들은 연산 비용이 크다는 문제가 있다
- Network Morphism 을 NAS에 써보자
	- 개념 : Neural Architecture를 바꾸면서도 뉴럴 네트워크의 기능을 유지하는 것
	- 이 Network Morphism을 이용해 NAS를 더 효율적으로 할 수 있도록 한다
- 본 논문은 BO를 Network Morphism에 적용해 NAS를 효율적으로 할 수 있도록 한다.
- 서치 공간을 효율적으로 탐색하기 위해 본 프레임워크가 개발하는 것은 다음과 같다 :
	- (1) Neural Network Kernel
	- (2) 트리구조의 acquisition function 최적화 알고리즘
	
## 1. Introduction
#### 딥러닝에서의 AutoML : NAS
- AutoML 목적은 머신러닝을 잘 모르는 사람들이 머신러닝 모델을 쉽게 사용할 수 있도록 하는 것이다
- 딥러닝의 경우, NAS(Neural Architecture Search)이 AutoML에서 효과적인 연산 툴로 자리 잡았다
	- NAS는 주어진 태스크와 데이터셋에 대해 좋은 뉴럴 네트워크 아키텍처를 찾는 것
- But, 존재하는 NAS 알고리즘은 **연산 비용이 크다**(computationally expensive)
	- NAS의 시간복잡도 : O(nt)
		- n : 서치 시 evaluate 되는 뉴럴 아키텍처 개수
		- t : 각 뉴럴 네트워크 평가하는 데 소요되는 시간 평균
- NAS 접근방법들에는 강화학습, gradient-based 방법들, 유전 알고리즘들이 있는데, 좋은 성능을 내기 위해 n의 값이 커야한다.
- 또한, 각 n 뉴럴 네트워크는 아예 밑바닥부터 학습하기에 매우 속도가 느리다 (trained from scratch)

#### 1. t를 줄이는 방법 : Network Morphism
- NAS에 Network Morphism을 적용하려는 시도가 있었다
- Network Morphism이란
	- 뉴럴 네트워크 아키텍처 기능을 유지하는 동시에 morph(조금씩 변형, to gradually change in appearance or form)하는 기술
- 이런 NM Operation을 이용해 학습된 뉴럴네트워크를 새로운 아키텍처로 바꿀 수 있다
	- NM Operation e.g. : 레이어 삽입, skip-conneciton 추가 등
- 단지 몇 개의 에폭만 더 필요하기 때문에 NM 사용시 NAS의 평균학습시간 t를 줄일 수 있다.

#### Network Morphism의 문제와 해결방법 제시
- NM 기반의 NAS 방법 적용시 **어떤 Operation을 적용할 것인지**가 가장 중요한 문제이다.
	- Network Morphism Operation Set 중 어떤 operation을 선택할지
- 그런데, NM 기반 NAS는 아직 충분히 효율적이지 못하다
	- 이유
		- 학습 데이터가 커야 하거나
		- 큰 서치 공간 (search space)에서 탐색하기 비효율적
- 가장 성능이 좋을 것으로 기대되는(promising) operation을 위해 BO를 사용하여 서치 공간 가이드에 활용한다

#### 2. n을 줄이는 방법 : Bayesian Optimization
- BO는 글로벌 최적화를 위한 블랙박스 함수를 탐색하는데 효율적이다
	- 이러한 탐색은 보통 expensive  to obtain 
	- e.g. hyperparameter tuning
	- 하이퍼파라미터 조합을 평가할 때 굉장히 expensive 한데, 이는 NAS와 유사
- BO의 유니크한 특성들을 참고하여 NM 이 학습한 뉴럴네트워크 수 n을 줄여 더 효율적으로 탐색할 수 있도록 해보자.

#### [NM 기반 NAS에 BO 적용시 challenges]
1. 뉴럴 네트워크 아키텍처에 GP 적용이 어렵다
- 가우시안 프로세스(GP)는 보통 유클리드 공간 함수의 확률분포를 학습하는 데 사용된다.
- BO 모델을 업데이트 하기 위해서는 탐색된 아키텍처와 성능을 기반으로 underlying GP가 학습된다
	- But 뉴럴 네트워크 아키텍처는 유클리드 공간에 있지 않다
	- 고정된 길이 벡터로 파라미터화 하기 어렵다

2. BO 적용시 NM이 트리구조임을 고려해야한다
- BO를 위해선 acquisitition function 이 최적화 되어야 아키텍처를 만들어 성능을 평가할 수 있다 (observe)
- But NM에서 중요한 것은 유클리드 공간에서 함수를 maximize하는 것이 아니다
- **트리 구조의 서치 공간에서 node를 찾는 것**
	- 이 때 node는 뉴럴 아키텍처를 의미
	- edge는 morph operation을 의미

3. NM Operation 결과는 복잡하다 (complicated)
- 한 레이어에서의 NM Operation은 중간 단의 출력 텐서의 shape을 바꿔버릴 수 있음
	- 이를 입력으로 받는 다음 레이어와 매칭이 안될 것
- 이러한 일관성 (consistency)를 유지하는 것도 하나의 해결할 문제

#### [편집거리를 이용]
- 앞에서 언급한 문제들을 해결하기 위해 **편집 거리 뉴럴네트워크 커널**이 만들어진다
- 편집거리 뉴럴네트워크 커널은 하나의 뉴럴 네트워크를 다른 네트워크로 바꾸기 위해 얼마나 많은 횟수의 operation이 필요한지 측정한다
- acquisition function optimizer
	- exploration 과 exploitation의 발란스를 맞추어주며
	- 여러 operation들에서 선택할 수 있도록 트리 구조 서치공간에 맞게 디자인 되었다
- 또한, graph-level NM은 layer-level의 NM을 기반으로 하는 뉴럴 아키텍처를 다룰 수 있도록 정의되었다
- 이렇게 하니 우리꺼가 SOTA보다 좋더라 ~		

## 2. Problem Statement

- 논문에서 NAS problem을 정의한 것을 간단히 적으면 다음과 같다 :
	- neural architecture search space 에서 최적의 뉴럴 네트워크를 찾는 것
	- search space는 쉽게 생각하면 가능한 뉴럴네트워크의 전체 집합이라고 생각하면 될 듯하다.
		- 먼저 특정 뉴럴네트워크 f에서 train 데이터에 대해 학습했을 때 loss를 minimize할 수 있는 최적의 하이퍼파라미터를 찾고, (공식 (2)에 해당)
		- 이 학습된 하이퍼파라미터를 적용한 뉴럴 네트워크로 valid 데이터를 평가했을 때 cost를 minimize 할 수 있는 뉴럴 네트워크를 찾는다 (공식 (1)에 해당)

## 3. Network Morphism Guided by BO

- 핵심을 다시 말하자면 **BO 알고리즘으로 뉴럴 아키텍처를 morphing 하여 서치 공간을 탐색**하는 것 (NAS하는 것) 이다.
- 전통적인(일반적인) BO는 3단계 - Update, Generation, Observation 의 반복으로 이루어진다
- NAS 맥락으로 보면 본 논문에서 제시하는 BO는 다음을 반복한다.
	- (1) Update
		- 존재하는(existing) 아키텍처 & 그 성능을 이용해 underlying 가우시안 프로세스 모델을 학습
	- (2) Generation
		- 다음 아키텍처를 생성한다.
	- (3) Observation
		- 생성된 뉴럴 아키텍처를 학습하여 성능 평가한다 (gain performance)
- BO를 이용해 NA를 morph하는데 3가지 핵심 문제들이 있는데(challenges) 다음과 같다.

### 3.1 Edit-Distance Neural Network Kernel for Gaussian Process

### 3.2 Optimization for Tree Structured Space

### 3.3 Graph-Level Network Morphism

### 3.4 Time Complexity Analysis


## 4. Auto-Keras

### 4.1 System Overview

### 4.2 Application Programming Interface

### 4.3 CPU and GPU Parallelism

### 4.4 GPU Memory Adaptation
