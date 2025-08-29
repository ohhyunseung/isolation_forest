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
```
python3 -m venv venv
source ./venv/bin/activate
(venv) ...# pip install -r requirements.txt
```

## 참고
NVIDIA GPU + CUDA 환경
```
pip install onnxruntime-gpu
```
> CUDA Toolkit 과 cuDNN 이 설치되어 있어야 정상 동작