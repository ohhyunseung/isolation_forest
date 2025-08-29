import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

X = np.array([[1], [2], [2], [3], [49], [100]])

# contamination 예상치 값 조절에 따른 다른 결과.... 몇 % 가 이상치 일가....
# value 49 가 이상치로 나올수도 있고 아닐수도 있고...
# random_state난 난수 발생 seed 값, 반복 실행 할때 일정한 결과가 나올수 있도록 함.
# model = IsolationForest(contamination=0.2, random_state=42)
model = IsolationForest(contamination=0.33, random_state=42)
model.fit(X)

df = pd.DataFrame(X, columns=["value"])
df["pred"] = model.predict(X)          # -1 이상치, 1 정상
df["score"] = model.decision_function(X)  # 낮을수록 이상치
print(df)
