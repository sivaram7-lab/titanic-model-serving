import requests
data = {"Pclass": 3, "Sex": "male", "Age": 25, "SibSp": 0, "Parch": 0, "Fare": 7.25, "Embarked": "S"}
r = requests.post("http://127.0.0.1:8000/predict", json=data)
print(r.status_code, r.json())
