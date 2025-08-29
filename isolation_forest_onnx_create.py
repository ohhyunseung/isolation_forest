import numpy as np
from sklearn.ensemble import IsolationForest
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

# 예시 데이터 생성
# 100개, feature 4 
# random 생성 특성 : 평균 0 표준편자 1 인 표준정규분포 사실상 범위 값은 -3~3 
X = np.random.randn(100, 4)
print(X)

# IsolationForest 학습
model = IsolationForest(random_state=42)
model.fit(X)

# ONNX 변환
# ONNX (Open Neural Network Exchange) 는
# 머신러닝/딥러닝 모델을 저장하고 교환하기 위한 표준 포맷
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    target_opset={'': 14, 'ai.onnx.ml': 3} # opset 버전 지정 : opset = ONNX 모델이 사용하는 연산자들의 버전
)

# 저장
with open("isolation_forest.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
