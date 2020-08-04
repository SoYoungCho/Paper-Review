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
		- feature vector ![](https://latex.codecogs.com/gif.latex?x_i), 대응되는 target value ![](https://latex.codecogs.com/gif.latex?y_i)
		- train data ![](https://latex.codecogs.com/gif.latex?D_%7Btrain%7D%20%3D%20%7B%28x_1%2C%20y_1%29%2C%20...%20%2C%20%28x_n%2C%20y_n%29%7D)
		- train data ![](https://latex.codecogs.com/gif.latex?D_%7Btest%7D%20%3D%20%7B%28x_%7Bn&plus;1%7D%2C%20y_%7Bn&plus;1%7D%29%2C%20...%20%2C%20%28x_%7Bn&plus;m%7D%2C%20y_%7Bn&plus;m%7D%29%7D)
		- these are drawn from the same underlying data distribution
			- (train & test 데이터는 동일한 모집단에서 추출되었다고 가정한다는 것)
		- 즉, AutoML 문제는 test 데이터의 예측값인 ![](https://latex.codecogs.com/gif.latex?%5Chat%7By%7D_%7Bn&plus;j%7D%2C%20...%2C%20%5Chat%7By%7D_%7Bn&plus;m%7D)을 자동으로 구하는 것
			- 이 때 Loss 는  
			![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bj%3D1%7D%5Em%20L%28%5Chat%7By%7D_%7Bn&plus;j%7D%2C%20y_%7Bn&plus;j%7D%29)
			- m개의 각 테스트 데이터 Loss의 전체 평균
- 위에서 **fixed computational budget** 표현이 있었는데, butget (![](https://latex.codecogs.com/gif.latex?b))란?
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
![](https://latex.codecogs.com/gif.latex?argmin%20%5Cfrac%7B1%7D%7BK%7D%20%5Csum_%7Bi%3D1%7D%5EK%20L%28A%5E%7B%28j%29%7D_%5Clambda%2C%20D%5E%7B%28i%29%7D%20_%7Btrain%7D%2C%20D%5E%7B%28i%29%7D_%7Bvalid%7D%29)
