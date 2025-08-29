from sklearn.ensemble import IsolationForest
import numpy as np

'''
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
'''

# 샘플 데이터
X = np.array([[1], [2], [2], [3], [100]])  # 100은 이상치로 가정

# 모델 훈련
clf = IsolationForest(contamination=0.25)
clf.fit(X)

scores = clf.decision_function(X)

# 예측 (-1 = 이상치, 1 = 정상)
print(clf.predict(X))
print(scores)