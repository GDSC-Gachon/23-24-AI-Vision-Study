### LeNet-5 이전의 흐름

---

- 퍼셉트론(Perceptron)의 등장 배경
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ab4044ad-b812-4adc-a02f-2d6e6317d1fd/ba7dd153-d63e-4dde-bf8e-b6f178b6c459/Untitled.png)
    
    - 컴퓨터가 인간처럼 생각하여 문제를 해결하길 바라는 아이디어에서 출발
    - 뉴런의 구조를 보면, 여러 가지 신호를 입력으로 받아 이 신호의 세기가 역치를 넘어서면 다른 뉴런으로 신호가 전달되는 흐름을 가지고 있다.
    - 이러한 구조를 모사하여 다음과 같은 퍼셉트론의 개념이 제안되었다.

- 퍼셉트론(Perceptron)
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ab4044ad-b812-4adc-a02f-2d6e6317d1fd/0919bf49-659c-4da2-8996-cad5cac8ed5c/Untitled.png)
    
    - 각 퍼셉트론은 가중치(w)를 가지고 있으며, 주어진 입력값(x)에 대한 곱을 모두 더한 후 바이어스(b) 값을 더한다.
    - 최종적으로 활성화 함수(activation function)에 이 값을 적용시켰을 때, 특정한 임계값을 만족한다면 그 값이 다음 퍼셉트론으로 전달된다.
    - 1957년 당시 퍼셉트론으로 문제를 해결하기 위해 하드웨어적인 구현에 대한 시도도 존재하였다.

- XOR 문제
    - 단일 퍼셉트론의 경우 결과값이 선형적이기 때문에, 모델의 예측은 “가장 훌륭한 예측선”을 그리는 것으로 설명할 수 있다.
    - AND, OR문제는 직선을 통한 분류가 가능했으나, 다음과 같은 XOR문제는 해결하지 못하여 난관에 봉착하였다.
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ab4044ad-b812-4adc-a02f-2d6e6317d1fd/ce154740-2337-48df-b513-1ab41547efea/Untitled.png)
        

- 다층 퍼셉트론(Multilayer perceptrons)
    - Perceptrons (1969) [by Marvin Minsky]에서 XOR문제를 해결하기 위한 방안으로 다층 퍼셉트론 모델을 제시하였다.
    - 하지만 이 책에서는 다층 퍼셉트론 모델을 학습할 방법론을 제시하지는 못했음
    - 책에서의 “No one on earth had found a viable way to train”, 즉 아무도 이를 학습시킬 방법을 찾을 수 없다는 표현 이후로 인공신경망 분야의 연구는 침체기를 맞이하게 된다.

- 역전파(Backpropagation) 방법론
    - 1986년 제프리 힌턴이 해당 방법론을 재발견했다고 알려짐
    - 인공신경망 분야에 다시 이목이 집중되기 시작함!
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ab4044ad-b812-4adc-a02f-2d6e6317d1fd/140d83a2-8bbc-4500-87c0-b2ef50652012/Untitled.png)
        
        - 순전파: 학습 데이터 셋에 대해서 예측
        - 역전파: 예측 결과와 정답 사이의 오차(error)를 역방향으로 전파하여 가중치를 수정
        - 다층 퍼셉트론을 학습시킬 수 있다.

- 합성곱 신경망(Convolutional Neural Networks)의 등장 배경
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ab4044ad-b812-4adc-a02f-2d6e6317d1fd/cdff87ab-687f-49ce-9247-d51b244e3cea/Untitled.png)
    
    - 고양이가 패턴을 인식할 때 시신경이 어떻게 반응하는지를 관찰
    - 고양이 시야의 한 쪽에 자극을 주었을 때 전체 뉴런이 아닌 특정 뉴런만이 활성화되었다.
    - 물체의 형태와 방향에 따라서도 활성화 되는 뉴런이 달랐다.
    - 동물의 시각 피질 안의 뉴런들은 일정 범위 안의 자극에만 활성화되는 근접 수용 영역(local receptive field)를 가지며 이 수용 영역들이 합쳐져 전체 시야를 이룬다는 것을 발견했다.
    - 이 결과를 토대로 Yann LeCun은 CNN의 시초라고 할 수 있는 LeNet 모델 구조를 발표한다. (1998)

### LeNet-5 모델 개요

---

- 가장 기본적인 CNN 모델 구조이다.
- 논문: [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)

### LeNet-5 모델 등장 배경

---

- 논문의 저자인 Yann LeCun이 손으로 적힌 우편 번호를 효율적으로 확인하기 위해 고안하였다.
- 문자 인식에서의 전통적인 방법
    - Hand-designed feature extractor
        - 입력으로부터 관련있는 정보만 수집하고 무관한 정보는 제거하므로 제한된 학습을 수행할 수 밖에 없다.
    - fully-connected multi-layer networks
        - 전체 네트워크가 완전 연결되어 있는 경우 학습해야 할 매개변수가 너무 많아진다.
        - 인접한 픽셀값을 공간적인 상관관계가 매우 큰데, 이러한 정보를 이용하지 못한다.

### CNN의 기본 개념 적립

---

- 수용 영역 (receptive field)
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ab4044ad-b812-4adc-a02f-2d6e6317d1fd/6d721f8a-a294-466f-8854-4e43511e4500/Untitled.png)
    
    - 앞선 고양이 실험에서와 같이 전체를 한 번에 보는 것이 아니라 작은 부분에 대한 feature를 결합하는 방식을 제안했다.
    - 학습 파라미터 수를 줄일 수 있다.
- 가중치 공유(shared weight)
    - feature map의 각 채널은 하나의 필터를 통과하여 만들어지기 때문에 동일한 가중치를 공유한다.
    - 학습 파라미터 수를 줄일 수 있다. (실제로 LeNet-5에는 340,908 connection이 존재하지만 60,000개의 trainable parameter만 존재)
- sub-sampling (현대의 pooling을 의미)
    - 논문에서는 한번 특징이 검출되면 위치 정보의 중요성이 떨어지며, 입력값에 따라 같은 특징이 나타나는 위치가 다를 가능성이 높기 때문에 잠재적으로 유해하다고 한다.
    - 따라서 단순히 해상도를 감소시키며, 이로 인한 손실은 더 많은 필터를 사용함으로써 더 다양한 특징을 추출하는 것으로 상호보완한다.

### LeNet-5 모델 구조

---

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ab4044ad-b812-4adc-a02f-2d6e6317d1fd/1fa1511b-2e2d-4992-8225-852973e87570/Untitled.png)

- Input
    - 32x32 손글씨 이미지
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ab4044ad-b812-4adc-a02f-2d6e6317d1fd/d48c3e91-7085-4975-84e0-85149cdf42e6/Untitled.png)
        
- Layer C1
    - 입력: `32x32` 이미지
    - Convolution 과정
        - 필터 크기: `5x5`
        - 필터 개수: `6개`
    - 출력: `28x28x6` feature map
- Layer S2
    - 입력: `28x28x6` feature map
    - Subsampling 과정
        - 필터 크기: `2x2`
        - 필터 개수: `6개`
        - 보폭(stride): `2`
        - 방식: average pooling
    - average pooling을 수행하는 필터에도 한 개의 훈련가능한 가중치와 바이어스가 존재한다. (시그모이드를 통해 활성화)
    - 출력: `14x14x6` feature map
- Layer C3
    - 입력: `14x14x6` feature map
    - Convolution 과정
        - 필터 크기: `5x5`
        - 필터 개수: `16개`
        - 이때 논문에서 설명한 특별한 방법을 사용한다. (더 다양한 특징을 추출하기 위함)
            
            ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ab4044ad-b812-4adc-a02f-2d6e6317d1fd/afa812ab-79d4-41b7-b9fc-3e5c46c413fe/Untitled.png)
            
            - 이때 연속된 3장, 4장 및 불연속된 4장의 필터 선택은 저자가 임의로 선택한 값
    - 출력: `10x10x16` feature map
- Layer S4
    - 입력: `10x10x16` feature map
    - Subsampling 과정
        - 필터 크기: `2x2`
        - 필터 개수: `16개`
        - 보폭(stride): `2`
        - 방식: average pooling
    - 출력: `5x5x16` feature map
- Layer C5 (Flatten)
    - 입력: `5x5x16` feature map
    - Convolution 과정
        - 필터 크기: `5x5x16`
        - 필터 개수: `120개`
    - 출력: 1x1x120
- Layer F6
    - 입력: `120개` 유닛
    - Fully-connected
    - 유닛 개수: `84개`
- Layer F7 (output layer)
    - 입력: 84개 유닛
    - Fully-connected
    - 유닛 개수: 10개
    - 10개의 출력에서 각각이 특정 이미지일 확률을 나타낸다.
- Loss function
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/ab4044ad-b812-4adc-a02f-2d6e6317d1fd/ff36b579-7dcd-4067-8b5a-e95b4c80a141/Untitled.png)

