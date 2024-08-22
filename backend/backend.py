from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
import os

app = FastAPI()

# Define the data model
class CustomerData(BaseModel):
    age: int
    monthly_spend: float
    contract_type: str
    tenure: int

# Load the model
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    try:
        model = joblib.load(model_path)
        print("Loaded existing model.")
        return model
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model not found. Please train the model first.")

# Load the model on startup
model = load_model()

# Prediction endpoint
@app.post("/predict")
def predict_churn(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    df = pd.get_dummies(df, columns=['contract_type'], drop_first=True)
    
    # Ensure all expected columns are present
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match the model's expected input
    df = df[expected_columns]
    
    prediction = model.predict(df)
    return {"churn": bool(prediction[0])}

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "OK"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)