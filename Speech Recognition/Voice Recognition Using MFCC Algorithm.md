# Voice Recognition Using MFCC Algorithm

- 본 논문은 카브 NLP & Speech 스터디에서 첫 주차로 진행한 논문이다.
- 내용은 MFCC 알고리즘이 어떻게 음성인식에 쓰이는지 절차를 정리하였다.
- 음성인식에 대한 베이스가 제로이기에 느낌을 파악하는 용도로 간단히 살펴보았다.

## Abstract
- 음성인식은 특정 사람의 목소리를 구별(identify)하기 위해 쓰이는 생체 인식 기술 중 하나
- 보안, 간편한 인증, 비용 절감에 효과적
- MFCC (Mel Frequency Cepstral Coefficients) 알고리즘은 음성인식을 위한 **피처 추출**하는데 많이 쓰인다.
	- 사람 개개인마다 unique 한 음성으로부터 coefficients(계수)를 만들어 내기 때문
- 키워드 : *Voice, Biometric technology, Feature Extraction, Authentication, MFCC*

## MFCC as a Voice Recognition Algorithm
음성인식 알고리즘을 위한  MFCC

- MFCC의 input : voice sample (음성 샘플)
- MFCC 과정을 거치면 특정 샘플에 대해 unique한 coefficients를 계산한다
- 논문은 MFCC 적용이 간편하다는 점이 장점이라고 말한다.
- 이제 MFCC 를 구하는 과정을 단계별로 살펴보자.

### Step 1. Pre-Emphasis
- Pre-Emphasis란, 잡음을 줄이기 위해 특정 주파수의 진폭을 높여 송신하는 조작이다.
- high-pass filter를 적용한다.
	- high-pass filter는 높은 주파수만 통과시킨다.
- high-pass filter를 거치고 나온 output signal은 주로 0.9 ~ 1.0의 값을 갖는다.
- Pre-Emphasis 목적?
	- 사람의 말고리는 일반적으로 저주파 성분의 에너지가 고주파보다 많은 경향이 있다.
	- 특히 이런 경향은 모음(ㅏ,ㅑ,ㅓ,ㅕ...) 에서 많이 관찰 된다.
	- 따라서 고주파 성분의 에너지를 올려주는 전처리 과정을 거치면 음성인식 성능을 개선할 수 있다.
	- 고주파의 포먼트(formant, 모음의 구성 음소)를 증폭(amplify)시킬 수 있다

### Step 2. Sampling and Windowing
![](https://www.researchgate.net/profile/Jason_Filos/publication/267779604/figure/fig3/AS:341695808983045@1458478037309/Frame-blocking-in-a-typical-MFCC-extraction-algorithm.png)

#### (1) Frame Blocking
- Input signal을 프레임 단위로 쪼개어 겹치게 하는 단계이다.
- 시그널은 15~20ms의 단위의 프레임으로 만들어지고, 절반씩 겹치게 overlap 한다.
- FFT 적용을 용이하게 하기 위해, 프레임 사이즈는 보통 (시그널의) sample point 개수에 제곱한 것과 같거나, 제곱한 값 정도까지 zero-padding 한다.
- 이렇게 Overlapping 하는 목적?
	- 프레임 내에서 **연속성**을 만들기 위함.
#### (2) Hamming Window
![](https://i.imgur.com/tHPxKTg.png)
- 위에서 만들어진 각 각의 프레임은 hamming window와 곱해진다.
- 목적?
	- 프레임의 첫 번째 , 마지막 point의 연속성을 유지하기 위해
- 다시 말해, continuity를 위해 각 프레임 내의 시그널을 hamming window와 곱한다.
- window 개념에 대해 조금 더 설명을 하면,
	- 긴 신호가 있을 때 신호의 일부에 초점을 두는 역할.
	- window는 음성 신호에 각 각 곱해지기에 convolution 효과를 줄 수 있다. (convolution :  서로 다른 두 신호를 합성하는 것)
	- window  종류에는 여러가지가 있는데 그 중 하나가 hamming window.
- hamming window는 관심 있는 영역에 일정 패턴의 대칭적인 weighting을 취함  (그래서 좌우 대칭인 종모양을 띈다)
	-  main lobe (중앙 가장 우뚝 솟은 부분)의 대역폭이 커지므로 window와 곱해 convolution하면 음성 신호의 스펙트럼 모양이 **smoothing** 되는 효과를 줄 수 있다.
	- [참고 블로그](https://deepinlife.tistory.com/10)

### Step 3. FFT (Fast Fourier Transform)
- 간단히 말하면 FFT는 시간, 혹은 신호를 주파수 단위로 변환하는 것이었다.
- 음성에서의 **음색(timbre)** = (energy distribution /  *frequncies*)이다.
- 따라서 FFT를 적용해 각 프레임의 크기 *주파수*를 구한다. (magnitude frequency)
- 프레임에 FFT를 적용하면, 프레임 내 시그널은 순환할 때 periodic(주기적), continuous(연속적)이라 할 수 있다.

### Step 4. Mel Filter Bank
![](https://blog.kakaocdn.net/dn/Qrpr1/btqwOA3zq1T/mLbjjVrkRG3QUvnAmYKpX1/img.png)
- Triangular Bandpass Filters
- 사람의 청각 기관은 고주파보다 **저주파 대역에서 더 민감**하다.
- 이런 인체학적 특징을 고려하여 주파수는 40개의 삼각형 모양의 주파수폭 필터와 곱해진다. (triangular band pass filters)
- 목적? 각 triangular band pass filter의 로그 에너지를 구하기 위함 (log energy)
- [참고 블로그](https://ahnjg.tistory.com/47)

### Step 5. Discrete Cosine Transform (DCT)
- 위 단계에서 나온 N 개의 triangular band pass filter들에 DCT를 적용, L개의 mel-scale ceptral coefficients (MFCC)를 구한다
- FFT는 시간 단위 -> 주파수 단위였다면,
- DCT는 다시 주파수 단위 -> 시간과 같은(time-like) 단위로 변환한다. (quefrency라고 함)
- 이렇게 DCT 로 얻은 피처들은 cepstrum 과 유사하기에 mel-scale ceptral coefficients 라 부른다. = MFCC
- MFCC는 단독적으로 음성인식을 위한 피처로 쓰일 수 있다.

### 정리
-  MFCC의 절차는 요약하면 다음과 같다.
1. Pre-Emphasis
	- 사람 말소리는 저주파 영역 에너지가 고르기에, 에너지를 고르게 하기 위해 고주파 부분 강조하는 high-pass filter 적용
  2. Sampling and Windowing
	 - 음성 시그널을 프레임 단위로 쪼갠 후 절반씩 겹치게 한다.
	 - 각 프레임에 hamming window 를 곱하여 smoothing 효과를 준다.
  3. 고속 푸리에 변환 (FFT)
	   - 시간 단위의 프레임들을 주파수 단위로 변환한다.
  4.  Mel Filter Bank
	   - 사람의 청각 기관은 고주파보다 저주파 영역에서 더 민감하기에, 주파수 크기 별 다른 폭의 필터를 곱해준다.
  5. Discrete Cosine Transform (DCT)
	  - 위 단계에서 나온 triangular band pass filter들은 이전에 FFT 를 통해 주파수 단위로 되어 있는데, 이를 다시 time-like 단위로 변환한다.
  

+ 추가 참고 블로그 : [ratsgo speech book](https://ratsgo.github.io/speechbook/docs/fe/mfcc)
