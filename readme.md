# 이상탐지 1편
isolation forest 를 알아보자.

```
- Forest means structure of trees.
- low depth is isolated.
- isolated value is anomal value.

                root (split @ 50)
                 /              \
        <=50 [1,2,2,3]        >50 [100] → isolated (depth=1)
            /       \
   <=2.5 [1,2,2]   >2.5 [3] → isolated
       /      \
<=1.5 [1]   >1.5 [2,2] → 계속 분할
```

## Start
학습 편이성 python이 좋다.
### Pthon
```
python3 -m venv venv
source ./venv/bin/activate
(venv) ...# pip install -r requirements.txt
```
### C++
https://github.com/microsoft/onnxruntime

ONNX의 사용 테스트 
 - ONNX는 표준이기 때문에 다른 언어 코드에서 사용가능
 - python 코드로 생성(isolation_forest_onnx_create.py), c++ 코드(isolation_forest_onnx_use.cpp)로 사용 테스트가 목적이다.
 - python 보다는 c++ 이 성능 우수하지 않을가
 - Onnxruntime  라이브러리의 버전 일치 하는 것이 좋겠다.
 - 컴파일 .vscode/tasks.json 참조

## 참고
NVIDIA GPU + CUDA 환경
```
pip install onnxruntime-gpu
```
> CUDA Toolkit 과 cuDNN 이 설치되어 있어야 정상 동작