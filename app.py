from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import joblib, pandas as pd

app = FastAPI(title="Titanic API")
model = joblib.load("model.pkl")

class Passenger(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: Literal["male", "female"]
    Age: float = Field(..., ge=0, le=120)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., ge=0)
    Embarked: Literal["S", "C", "Q"]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(p: Passenger):
    df = pd.DataFrame([p.model_dump()])
    pred = int(model.predict(df)[0])
    return {"prediction": pred, "result": "Survived" if pred == 1 else "Not Survived"}
