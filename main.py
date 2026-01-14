# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("sales_forecast_model.joblib")

app = FastAPI(title="Sales Forecast API")

# Schema to Request
class PredictRequest(BaseModel):
    lag_1: float
    lag_7: float
    rolling_7: float
    day_of_week: int
    month: int

# Endpoint to test
@app.get("/")
def read_root():
    return {"message": "Sales Forecast API is running!"}

# Endpoint to predect
@app.post("/predict")
def predict(request: PredictRequest):
    X_pred = pd.DataFrame([[
        request.lag_1,
        request.lag_7,
        request.rolling_7,
        request.day_of_week,
        request.month
    ]], columns=["lag_1", "lag_7", "rolling_7", "day_of_week", "month"])
    
    y_pred = model.predict(X_pred)[0]
    return {"predicted_sales": float(round(y_pred, 2))}
