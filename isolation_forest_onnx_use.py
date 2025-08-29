import numpy as np
import onnxruntime as rt

# 1. ONNX 모델 로드
# isolation_forest.onnx 는 랜덤함수에 의한 정규분포로 각 feature 값이 생성됬을 것 이므로,
sess = rt.InferenceSession("isolation_forest.onnx")

# 2. 입력/출력 이름 확인
input_name = sess.get_inputs()[0].name
output_names = [out.name for out in sess.get_outputs()]

print("입력 이름:", input_name)
print("출력 이름:", output_names)

# 3. 새로운 입력 (feature 4개짜리)
# 대체로 0 주의 값을 정상으로 볼 수 있으므로, 
# 비정상(-1)을 원하면 0에서 많이 벚어난 feature 값으로 해보되,
# -3 ~ 3 안쪽이라면, 4개의 feature 값에 의한 판단은 예상처럼 -1이 아닐수 있다.
new_input = np.array([[0.5, 1.2, -0.3, 1.1]], dtype=np.float32)
#new_input = np.random.randn(5, 4).astype(np.float32)

# 4. 모델 실행
pred_onx = sess.run(output_names, {input_name: new_input})

# 5. 결과 출력
print("입력 feature :", new_input)
for name, val in zip(output_names, pred_onx):
    print(f"{name} => {val}")
