# LeNet-5

### Lenet-5
이미지 분류 CNN(신경망 네트워크)의 시초로 글자 인식을 위한 신경망으로 총 7개의 층으로 이루어져 있으며, 각 층은 Convolutional layer와 Subsampling layer으로 구성되어 있다.

### architecture
![tQGDtMp.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/6c91ff93-899d-488e-9d3c-0fae0ceb24c4/0af828b3-5b86-4fbd-b098-e85d79cb8068/tQGDtMp.jpg)
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/6c91ff93-899d-488e-9d3c-0fae0ceb24c4/3e1282c9-321a-451c-abb7-e86c5b079f3d/Untitled.png)

### layer순서
**input> C1 > S2 > C3 > S4 > C5 > F6 > output**

convolution > 필터를 통해 각 이미지의 feature를 추출해서 feature map(feature bank)에 저장  
subsampling > 사이즈를 줄여주는 작업 

### Trainable parameter 계산 규칙
<aside>
💡 (가중치* 입력맵개수 + 바이어스) * 특성맵개수
</aside>

(필터가 5*5일때 가중치가 곧 mask 이므로 5x5가 가중치가 된다.)  

- 의문점 : 왜 훈련가능한 가중치를 곱해주고 바이어스를 더해줘야할까?  
    단순 풀링만 하는 것이 아니라 가중치와 바이어스가 sigmod의 비활성도를 조절해주는 역할을 한다.
  
### **C1 Layer(Convolution)**

- 여기서 필터는 5*5 크기의 필터를 사용하며 입력 이미지와 합성곱 연산을 수행하며 여러 필터가 사용되어서 서로 다른 특징들을 감지한다. 입력 이미지의 5 x 5 영역이 각 feature map의 한 유닛으로 연결되도록 한다.  
- 필터의 가중치, 특성은 역전파 알고리즘을 사용하여 조정되며 중요한 패턴을 추출하는 데 사용되는 필터를 학습한다. 따라서 초기 필터는 input값에 따라 달라 질 수 있다.  

- 의문점:  어떻게 역전파 알고리즘을 사용하여 필터가 조정될까?    
    
    역전파(Backpropagation) 알고리즘은 신경망에서 가중치와 필터를 조정하는 과정을 학습한다.     
    1. 순방향 전파 (Forward Propagation):
        - 입력 데이터가 신경망을 통과하여 예측을 생성하면 예측과 실제 목표값 사이의 오차를 계산함
    2. 역방향 전파 (Backward Propagation):
        - 출력 오차를 역전파하여 신경망 내부의 각 뉴런 및 가중치에 대한 그래디언트(기울기)를 계산하고 필터를 조정하기 위한 필터 그래디언트를 계산함.
    3. 가중치 및 필터 업데이트:
        - 계산된 그래디언트를 사용하여 가중치 및 필터를 업데이트하는데 이때 경사하강법 최적화 알고리즘을 사용함업데이트합니다.
        - 필터의 각 원소를 현재 값에서 그래디언트에 학습률(learning rate)을 곱한 값을 빼는 방식으로 조정함.
  

### **S2 Layer(subsampling)**

여기서는 2 x 2필터로 downsampling을 하여  average pooling을 취해주었다. 2 x 2필터가 픽셀 하나를 생성하여 사이즈가 절반인 피처 맵을 출력한다.
이는 공간 해상도를 줄이고 계산 효율성을 향상 시킨다.

    

### **C3 Layer(Convolution)**

![ASzysLD.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/6c91ff93-899d-488e-9d3c-0fae0ceb24c4/56b61989-e21a-4829-80cb-db4a5997077c/ASzysLD.png)

참고자료:[https://velog.io/@lighthouse97/LeNet-5의-이해](https://velog.io/@lighthouse97/LeNet-5%EC%9D%98-%EC%9D%B4%ED%95%B4)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/6c91ff93-899d-488e-9d3c-0fae0ceb24c4/a8466c26-1e3b-4864-9e55-08a6c8d62565/Untitled.png)

계산해보자: 14-5+1=10> 10*10

c3는 c1와 다르게 s2의 여러 피처맵을 한번에 참조해서 c3의 한 유닛으로 연결된다.
테이블은 S2의 어떤 feature map이 C3의 어떤 feature map과 연결되어 있는지 보여준다.

- 의문점: 왜 s2의 모든 피처맵이 c3의 모든 feature map과 연결되지 않을까?  
    바로 network의 대칭을 파괴하기 때문! 같은 weight을 가지더라도 connection이 다르면 서로 다른 피처맵이 생성됨.
    그리고 전부 연결되게 되면 대칭이 생겨 같은 필터들이 만들어져 비효율적

### **S4 Layer(subsampling)**

16장의 10 x 10 특성 맵에 대해서 서브샘플링을 진행해 16장의 5 x 5 피처 맵으로 축소시킨다.(2 x 2 필터, stride 2)

### **C5 Layer(convolution)**

16장의 5 x 5 특성맵을 120개의 5 x 5 x 16 사이즈의 필터와 컨볼루션 해준다. 결과적으로 120개의 1 x 1 피처맵이 산출된다


### **F6 Layer(Fully-Connected)**
 F6 layer는 84개의 unit이 있고 C5와 fully-connected 되어 있다.
 F6 까지는 기존의 신경망과 같이 가중치 합에 bias를 더한 뒤 이를 활성함수에 통과시켜 출력 값으로 사용한다. 
 이 때, 사용된 함수는 하이퍼벡터 Hyperbolic Tangent 함수이다.

### **ouput Layer(Fully-Connected)**
 마지막의 output layer는 Euclidean Radial Basis Function(RBF)으로 구성되어 있고 최종적으로 이미지가 속한 클래스를 알려준다. 
 f6 layer의 weight들은 +1 아니면 -1인데 이는 sigmoid의 최대 곡률과 일치한다.
 10개 class로 구분하여 출력해서 각각 특정 이미지일 확률을 나타낸다.


