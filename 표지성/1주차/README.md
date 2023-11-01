# LeNet

LeNet-5은 1998년 Yann LeCun의 논문 'Gradient-Based Learning Applied to Document Recognition' 에 담겨있는 CNN 신경망의 구조

## LeNet의 등장 배경

1. Yann LeCun이 손으로 적힌 우편 번호를 전통적인 방법보다 효율적으로 확인하기 위해 고안된 CNN 구조
2. 패턴 인식에서 이용되는 전통적인 모델은 hand-designed feature extractor로 특징을 추출 → fully-connected multi-layer networks를 분류기로 사용
3. 여러가지 문제점이 발생
   1. Hand-designed feature extractor는 제한된 특징만 추출<br>
           관련있는 정보만 수집하고 무관한 정보는 제거하는데, feature extractor에 의해 추출된 정보만 가지고 classifier의 학습이 진행되므로 학습에 제한이 있다. LeCun은 feature extractor 그 자체에서 학습이 이루어져야 한다고 생각했다.

   2. 너무 많은 매개변수를 포함<br>
        이미지를 FC로 전환해 학습하는 방식은 너무 많은 parameter를 포함한다.

   3. 입력값의 topology가 완전히 무시<br>
        입력값의 Topology가 완전히 무시된다. 이미지는 기본적으로 2D 구조를 가지고 있는데, 이는 공간적으로 매우 큰 상관관계가 있으며 FC는 이미지를 일렬로 펼치기 때문에, 이런 공간적인 관계를 완전히 무시하게 된다.
        
## LeNet-5의 구조
![Untitled](https://github.com/pjs990301/23-24-AI-Vision-Study/blob/main/%ED%91%9C%EC%A7%80%EC%84%B1/1%EC%A3%BC%EC%B0%A8/figure/fig1.png?raw=true)

1. 7-layer
2. 3개의 Convolution(C1,C3,C5)
3. 2개의 Subsampling(S2,S4)
4. 1개의 Fully-Connected(F6)
5. 1개의 RBF(Output)
6. 입력 데이터는 1x32x32 사이즈의 Normalize된 흑백 이미지
7. 손실 함수는 MSE(Mean Squared Error)를 사용

## LeNet-5의 각 계층 설명

💡 훈련해야할 파라미터 개수: (가중치 + 바이어스) * 특성맵개수

### C1 Layer(Convolution)

1. 입력 이미지(1x32x32)를 입력받아 6개의 5x5 필터와 Convolution 연산(stride=1, padding=0)을 진행 
2. 그 결과 6장의 feature map(6x28x28)이 출력
3. 파라미터의 수 : 5 x 5 x 6 + 6 = 156

### Layer(Subsampling)

1. 6장의 feature map(6x28x28)을 입력받아 2x2 average pooling 연산(stride=2, padding=0)을 진행
2. 그 결과 6장의 feature map(6x14x14)이 출력
3. 출력된 feature map은 sigmoid 연산
4. 파라미터의 수 : (1 + 1) x 6 = 12

### C3 Layer(Convolution)

1. 6장의 feature map(6x14x14)을 입력
2. 16개의 5x5 필터와 Convolution 연산(stride=1, padding=0)을 진행
3. 그 결과 16장의 feature map(16x10x10)이 출력
4. 파라미터의 수
   1. 첫번째그룹 ⇒ (가중치 * 입력맵개수 + 바이어스) * 특성맵 개수 = (5 * 5 * 3 + 1) * 6 = 456
   2. 두번째그룹 ⇒ (가중치 * 입력맵개수 + 바이어스) * 특성맵 개수 = (5 * 5 * 4 + 1) * 6 = 606
   3. 세번째그룹 ⇒ (가중치 * 입력맵개수 + 바이어스) * 특성맵 개수 = (5 * 5 * 4 + 1) * 3 = 303
   4. 네번째그룹 ⇒ (가중치 * 입력맵개수 + 바이어스) * 특성맵 개수 = (5 * 5 * 6 + 1) * 1 = 151

![Untitled](https://github.com/pjs990301/23-24-AI-Vision-Study/blob/main/%ED%91%9C%EC%A7%80%EC%84%B1/1%EC%A3%BC%EC%B0%A8/figure/fig2.png?raw=true)

### S4 Layer(Subsampling)

1. 6장의 feature map(16x10x10)을 입력
2. 2x2 average pooling 연산(stride=2, padding=0)을 진행
3. 그 결과 16장의 feature map(16x5x5)이 출력
4. 파라미터의 수 : (1 + 1) x 16 = 32

### C5 Layer(Convolution)

1. 6장의 feature map(16x5x5)을 입력
2. 120개의 5x5 필터와 Convolution 연산(stride=1, padding=0)을 진행
3. 그 결과 120장의 feature map(120x1x1)이 출력
4. 파라미터의 수 : (5 x 5 x 16) x 120 + 120 = 48120

### F6 Layer(Fully-Connected)

1. 120개의 노드를 입력
2. 84개의 노드를 출력하는 Fully-connected 연산을 진행
3. 파라미터의 수 : (120 x 84) + 84 = 10164

<aside>
💡 84라는 숫자는 출력값인 ASCII set이 7x12의 bitmap이어서 설정
</aside>

### Output Layer

1. 84개의 노드를 입력
2. 10개의 노드를 출력하는 RBF(Euclidean Radia Basis Function unit) 연산을 진행
3. 최종 파라미터의 수 : C1 + S2 + C3 + S4 + C5 + F6 = 156 + 12 + 1516 + 32 + 48120 + 10164 = 60000

## Reference

[1] [https://velog.io/@lighthouse97/LeNet-5%EC%9D%98-%EC%9D%B4%ED%95%B4](https://velog.io/@lighthouse97/LeNet-5%EC%9D%98-%EC%9D%B4%ED%95%B4)    
[2] [https://mingyu6952.tistory.com/entry/LeNet-5-Pytorch%EB%A1%9C-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0](https://mingyu6952.tistory.com/entry/LeNet-5-Pytorch%EB%A1%9C-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0)    
[3] [https://deep-learning-study.tistory.com/368](https://deep-learning-study.tistory.com/368)    
[4] [https://wikidocs.net/137250](https://wikidocs.net/137250)    
[5] [https://github.com/juni5184/Paper_review/blob/main/(pytorch)lenet-5.ipynb](https://github.com/juni5184/Paper_review/blob/main/(pytorch)lenet-5.ipynb)
[6] [https://mingyu6952.tistory.com/entry/LeNet-5-Pytorch%EB%A1%9C-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0](https://mingyu6952.tistory.com/entry/LeNet-5-Pytorch%EB%A1%9C-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0)
