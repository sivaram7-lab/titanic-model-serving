import pandas as pd, joblib
model = joblib.load("model.pkl")
unseen = pd.read_csv("unseen_data.csv")
batch = unseen.iloc[1:21].copy()
batch["Prediction"] = model.predict(batch)
batch.to_csv("batch_predictions.csv", index=False)
print(batch)
