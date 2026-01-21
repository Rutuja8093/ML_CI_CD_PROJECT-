from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

X = np.array([[1,2],[2,3],[3,4],[4,5]])
y = [0,0,1,1]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model trained successfully")
