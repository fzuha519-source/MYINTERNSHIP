from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load the saved model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Create the FastAPI app
app = FastAPI()

# Define what input data looks like
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

# Home route
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running! ✅"}

# Prediction route
@app.post("/predict")
def predict(data: CustomerData):
    # Convert input to array
    input_data = np.array([[
        data.gender, data.SeniorCitizen, data.Partner,
        data.Dependents, data.tenure, data.PhoneService,
        data.MultipleLines, data.InternetService, data.OnlineSecurity,
        data.OnlineBackup, data.DeviceProtection, data.TechSupport,
        data.StreamingTV, data.StreamingMovies, data.Contract,
        data.PaperlessBilling, data.PaymentMethod,
        data.MonthlyCharges, data.TotalCharges
    ]])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return {
        "churn_prediction": int(prediction),
        "result": "Will Churn ❌" if prediction == 1 else "Will Stay ✅",
        "churn_probability": round(float(probability), 4)
    }