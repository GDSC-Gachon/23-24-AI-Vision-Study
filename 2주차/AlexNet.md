<img width="624" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/9e62059f-3454-4ff7-8654-1a9f560f94c1">

## 사용 Dataset

ImageNet의 subset. 

1,200,000 train / 50,000 val / 150,000 test

1000 classes

- optimizer: SGD
- momentum: 0.9
- weight decay: 5e-4
- batch size: 128
- learning rate: 0.01
- adjust learning rate: validation error가 현재 lr로 더 이상 개선 안되면 lr을 10으로 나눠줌. 0.01을 lr 초기 값으로 총 3번 줄어듦
- epoch: 90

## Features

<details>
<summary>활성함수로 ReLU 사용.</summary>
<div markdown="1">
    <img width="412" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/00f70bf0-2063-4802-9bb6-2085a5250370">

    점선: tanh / 실선: ReLU
    
    같은 정확도일 때 tanh보다 6배 정도 빠르다고 한다.
    
    이후 ReLU 사용이 보편화되었다.

</div>
</details>

<details>
<summary>LRN - Local Response Normalization 사용.</summary>
<div>

    → 활성화 함수를 적용하기 전에 적용하여 결과값에서 더 좋은 결과 도출	
    
    LRN은 학습할 이미지 영역의 피쳐 맵에서 픽셀 값을 정사각형으로 정규화하는 기법입니다.
    
    lateral inhibition - 강한 뉴런의 활성화가 다른 뉴런의 활동을 억제시키는 현상.
    
    <img width="303" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/05b83ae4-635f-4822-b9cd-9b96c79a1064">

    
    ex) 검은색 사각형을 보고 뉴런들이 강하게 반응하여 흰색 부분에 회색이 조금 보이게 됨.
    
    이처럼 neural network의 feature에서 한 cell이 굉장히 강한 값을 가지게 된다면 근처의 값이 약하더라도 convolution 연산을 거쳤을 때 강한 feature를 가지게 된다. 그러면 training dataset에만 feature가 크게 반응하므로 overfitting이 잘 발생할 수 있다.
    
    이를 방지하기 위해 그 주위 혹은 같은 위치에 다른 channel의 filter들을 square-sum하여 한 filter에서만 과도하게 activate하는 것을 방지함.
    
    <img width="420" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/752f88c9-61c0-4ad3-844c-c441f1fbf2eb">

    
    AlexNet에서 처음 도입. 
    
    최근에는 Batch Nomalization 사용.

</div>
</details>

<details>
<summary>네트워크 분할 사용.</summary>
<div markdown="1">

    GTX580, 메모리 3GB 두 개를 병렬로 사용했다.
    
    - 메모리 제한이 있었기 때문이며, AlexNet 이후에는 컴퓨팅 성능이 향상됨에 따라 이러한 구조를 사용하지 않는다.
    
    <img width="259" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/1e93a3f9-2d88-4748-82b7-ae206d5190ac">

    
    첫 번째 conv. layer의 모습.
    
    위가 GPU-1이고 아래가 GPU-2인데, GPU-1에서는 컬러와 상관없는 48개의 필터를 학습했고, GPU-2에서는 컬러와 관련된 48개의 필터를 학습했다.
    
    → 1st conv layer의 개수는 (48 + 48) = 96개!

</div>
</details>

<details>
  <summary>Overlapping pooling</summary>
  <div markdown="1">

    stride를 커널 사이즈보다 작게 하여 overlap되도록 구성했다.
    
    <img width="272" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/0dd369cf-4a6c-44cd-ad79-03765b520993">

    
    이렇게 중첩시킴으로 인해 top-1, top-5 에러율을 줄이는데 효과가 있었다.
    
  </div>
</details>

<details>
  <summary>Dropout</summary>
  <div markdown="1">
    verfitting을 막기 위해 사용.
    
    FC layer에서 50% 확률의 Dropout 적용
    
    training 시에만 사용하고, test시에는 사용하지 않음.
  </div>
</details>
    
- Data Augmentation 사용. 데이터 양을 2048배 늘림

## Architecture

<img width="636" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/7a25f84e-027f-4a15-bad4-1caabe7d1364">


- 256x256 이미지를 flip 및 random crop 하여(augmentation) 227x227 image size를 만들어 사용
- 5개의 conv. layer는 두 개의 GPU에 필터 개수를 분할하여 학습.
- 첫번째 FC layer에서 총 256개 feature map을 flatten시킨다.

<img width="401" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/c86b991c-6fed-4fcf-a6f3-54babeea06ad">
<img width="438" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/91236ba1-4314-4e61-855a-61783c0d2ffb">
<img width="572" alt="image" src="https://github.com/9-coding/23-24-AI-Vision-Study/assets/127665166/ecb3d909-c160-47c2-8385-cb09f55581bf">


## References

- https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
- https://deep-learning-study.tistory.com/376
- https://www.datamaker.io/blog/posts/34
- https://wjunsea.tistory.com/92
- https://curaai00.tistory.com/4
- [https://bskyvision.com/entry/CNN-알고리즘들-AlexNet의-구조](https://bskyvision.com/entry/CNN-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EB%93%A4-AlexNet%EC%9D%98-%EA%B5%AC%EC%A1%B0)
