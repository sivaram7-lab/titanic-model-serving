import pandas as pd, joblib, time
model = joblib.load("model.pkl")
unseen = pd.read_csv("unseen_data.csv")
for i in range(21, 31):
    r = unseen.iloc[[i]]
    print(f"Record {i-20}: prediction =", model.predict(r)[0])
    time.sleep(1)
