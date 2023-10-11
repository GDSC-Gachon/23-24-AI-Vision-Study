## Intro

### 왜 CNN이 탄생했을까?

<img width="673" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/6a59afd0-5f36-4457-bd8e-8f52112e184d">


**fully connected neural net 단점**

- 클래스의 평균적인 이미지와 동떨어져있는 이미지의 경우 잘 표현되지 않음.
- Test time에서의 성능이 매우 떨어질 수 있음.
- 이미지의 변형에 취약함.

**CNN - Convolutional Neural Network** 

- 국부적인 영역에 대한 연결만 고려한 Locally Connected Layer를 바탕으로 구성.
- 파라미터를 재활용(Parameter sharing)할 수 있기 때문에 파라미터의 수를 획기적으로 줄일 수 있음.
    
    →  overfitting을 방지하는 효과.
    
- 지역적인 특징을 추출하기 때문에 약간 변형된 이미지를 입력하더라도 올바르게 분류할 수 있음.

하나의 특징을 뽑기 위해 이미지의 전체를 검사하는 fully connected neural net에서 벗어나, 이미지의 공간적인 특성을 고려하여 국부적인 영역만 connection을 고려하는 locally connected neural net 채용.

→ 필요한 파라미터가 획기적으로 감소.

→ connection을 공유해서 전영역을 순회하며 feature를 추출함.

→ overfitting 방지까지.

## Convolution
<img width="670" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/e4d9387c-bd4f-4e0b-9cb1-a2e09e599e33">
<img width="672" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/ecff1da3-dad2-4285-8175-ade1d10e0a86">
<img width="671" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/69e74d3f-b685-4eeb-bbb8-49436b30da11">
<img width="673" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/e44f1427-f786-4fcd-b453-ed269830de17">
<img width="186" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/a1f58e3d-5d72-446a-ade6-4c7c1aa145a7">


## Architecture
<img width="662" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/984efadf-e491-46cc-bb48-8efef9aed89a">

- input: 32x32 image
    
    실제 문자 이미지는 28x28 이지만 corner나 edge 같은 특징이 receptive field의 중앙 부분에 나타나길 원하기 때문에 이렇게 설정함.
    
- C : Convolution layer  |  S: Sub-sampling (average pooling)  |  Fully-connected layer

input size → process → output size

- C1: 32x32x6 → 5x5 filter 6개를 사용해 28x28 size feature map 6개 생성 → 28x28x6
- S2: 28x28x6 → 2x2 filter 6개를 사용해 14x14 size feature map 6개 생성 → 14x14x6
- C3: 14x14x6 → 5x5 filter 16개를 사용해 10x10 size feature map 16개 생성 → 10x10x16
- S4: 10x10x16 → 2x2 filter 16개를 사용해 5x5 크기의 feature map 16개 생성 → 5x5x16
- S5: 5x5x16 → 5x5 filter 120개를 사용해 1x1 크기의 feature map 120개 생성 → 1x1x120
- F6:  tanh activation function. 120 → 1x1 feature map 84개 생성 → 84
- F7: RBF(euclidean Radia Basis Function unit).
       84 → outptlayer → 10

- subsampling 이유
    
    추론을 하는데에 있어서 위치정보가 과다해지면 오히려 독이 되기 때문이다.
    
    그래서 위치이동, 회전, 부분적인 변화, 왜곡에 강한 인식력을 키우기 위해 subsampling이 중요하다.

    
    위 그림에서 보다시피 A의 기본 폰트에서 작은변화를 주었을 때, 위치 이동을 하였을 때도 사람은 이를 여전히 A라고 인식한다.
    
    이런 모든 변형의 경우의 수를 학습하는건 너무 비효율 적이여서 subsampling을 사용한다. subsampling을 통해서 입력 데이터의 크기를 줄이면 두드러지는(강한) 특징만 남기고 자잘한 변화들은 사라지는 효과를 얻게 된다.
    
    반복적인 subsampling을 거치면 형태가 거의 유사해진다
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/a99d2330-d359-40b6-b12f-ae3f90c9f4e5/c38cb2b5-c6af-436f-8e08-bb0a4d1c02be/Untitled.png)
    
    다시 아키텍쳐를 살펴보면, convolution layer를 통해 입력데이터(input)로 부터 중요한 특징을 추출한 후 subsampling layer에서 입력데이터의 크기를 줄인다. 이후 다시 convolution과 subsampling을 통해 크기를 5x5까지 줄이면서 점점 위치나 크기 변화 등에 강한 특징만 남기게 된다.
    

## Result
<img width="648" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/d63e762a-e79b-41c0-9515-376722b3d394">
<img width="663" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/d2b5d2a3-6df6-44b6-bdfe-25da4bdf246f">

### References
- https://www.boostcourse.org/ai340/lecture/1462953?isDesc=false
- https://eehoeskrap.tistory.com/704
- https://eunhye-zz.tistory.com/10
- https://brave-greenfrog.tistory.com/45?category=1082810
