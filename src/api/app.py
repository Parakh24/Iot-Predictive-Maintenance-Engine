import os
import joblib
import pandas as pd
import numpy as np
import json
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
from typing import Dict, Any

# Initialize FastAPI app
app = FastAPI(title="IoT Predictive Maintenance Engine API")

# Global variables for model assets
model = None
scaler = None
feature_names = None

def load_model_assets():
    """Load model, scaler, and feature names from the latest training artifacts."""
    global model, scaler, feature_names
    
    # Define paths - targeting the artifacts generated in Week 3 SHAP integration
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    artifacts_dir = os.path.join(base_dir, "src", "explain", "shap_outputs")
    
    model_path = os.path.join(artifacts_dir, "xgboost_model.joblib")
    scaler_path = os.path.join(artifacts_dir, "scaler.joblib")
    features_path = os.path.join(artifacts_dir, "feature_names.json")
    
    try:
        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        
        print(f"Loading scaler from {scaler_path}...")
        scaler = joblib.load(scaler_path)
        
        print(f"Loading feature names from {features_path}...")
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
            
        print("All model assets loaded successfully.")
    except Exception as e:
        print(f"Error loading model assets: {e}")
        # Not raising error here to allow app to start, but /predict will fail

# Load assets on startup
load_model_assets()

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    status: str
    inference_time_ms: float

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "IoT Predictive Maintenance API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: Dict[str, Any]):
    """
    Endpoint to predict machine failure from sensor data.
    """
    start_time = time.time()
    
    if model is None or scaler is None or feature_names is None:
        raise HTTPException(status_code=503, detail="Model assets not loaded")
    
    try:
        # Validate and prepare input data
        input_data = data
        
        # Sanitize keys to match training data format (replace spaces and brackets with underscores)
        sanitized_data = {}
        for k, v in input_data.items():
            new_k = str(k).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_').replace(' ', '_')
            sanitized_data[new_k] = v
            
        # Ensure all required features are present
        missing_features = [f for f in feature_names if f not in sanitized_data]
        if missing_features:
             # Try to fill missing with 0 for robustness? Or fail?
             # Let's try to find if the raw key exists but sanitization failed? 
             # For now, strict check on sanitized keys
             raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")
            
        # Create DataFrame ensuring correct order
        df = pd.DataFrame([sanitized_data])
        df = df[feature_names]
        
        # Scale data
        df_scaled = scaler.transform(df)
        
        # Predict
        prediction = int(model.predict(df_scaled)[0])
        probability = float(model.predict_proba(df_scaled)[0][1])
        
        # Determine status
        status_map = {0: "Normal Operation", 1: "Potential Failure Detected"}
        status_text = status_map.get(prediction, "Unknown")
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return {
            "prediction": prediction,
            "probability": probability,
            "status": status_text,
            "inference_time_ms": round(inference_time, 2)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
