import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 1. 로그 샘플 데이터 (API Gateway 로그 느낌...)
# 예: timestamp, response_time, status_code, request_size, user_id 등
data = [
    {"response_time": 120, "status_code": 200, "request_size": 512},
    {"response_time": 80, "status_code": 200, "request_size": 256},
    {"response_time": 5000, "status_code": 200, "request_size": 1024},  # 이상치 후보
    {"response_time": 100, "status_code": 500, "request_size": 300},   
    {"response_time": 90, "status_code": 200, "request_size": 280},
    {"response_time": 110, "status_code": 200, "request_size": 290},
]

df = pd.DataFrame(data)

# 2. 숫자형 피처 선택 (status_code도 포함할 수 있음)
features = ["response_time", "status_code", "request_size"]
X = df[features]

# 3. 스케일링 (값들의 범위를 맞춤)
# response_time 이 120, 80, 5000, 100, 90, 110 인데
# 이 값들을 평균 0, 분산 1 로 변환  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. IsolationForest 적용
model = IsolationForest(contamination=0.2, random_state=42)
df["anomaly"] = model.fit_predict(X_scaled)

# 5. 결과 해석: -1은 이상치, 1은 정상
print(df)
