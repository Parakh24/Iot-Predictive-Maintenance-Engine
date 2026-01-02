import os
import joblib
import pandas as pd
import shap
import numpy as np

# Resolve Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_PATH = os.path.join(BASE_DIR, "modeling", "models", "xgboost_pipeline.joblib")

# 1. Load the full pipeline
pipeline = joblib.load(PIPELINE_PATH)
model = pipeline.named_steps.get("model") or pipeline.named_steps.get("classifier")

# 2. Setup SHAP Explainer
# We use a background of zeros with the correct 8032 shape
def model_predict_func(X):
    return pipeline.predict_proba(X)[:, 1]

background = pd.DataFrame(
    np.zeros((1, len(pipeline.feature_names_in_))),
    columns=pipeline.feature_names_in_
)
explainer = shap.Explainer(model_predict_func, background)

def predict_failure(sensor_json: dict):
    # FIX: Force all inputs to float to avoid 'str' comparison errors
    clean_data = {k: float(v) for k, v in sensor_json.items()}
    input_df = pd.DataFrame([clean_data])

    # FIX: Shape Mismatch (Create 8032 columns)
    full_features = pd.DataFrame(
        np.zeros((1, len(pipeline.feature_names_in_))), 
        columns=pipeline.feature_names_in_
    )

    # Fill the 4 user values into the 8032-column template
    for col in input_df.columns:
        if col in full_features.columns:
            full_features[col] = input_df[col]
    
    X = full_features.astype(float)

    # 3. Predict Probability
    failure_prob = pipeline.predict_proba(X)[0][1]

    # 4. Generate SHAP Summary (Local Explanation)
    shap_results = explainer(X)
    
    shap_summary = {}
    for feature in input_df.columns:
        if feature in full_features.columns:
            idx = list(full_features.columns).index(feature)
            val = abs(shap_results.values[0][idx])
            shap_summary[feature] = round(float(val), 4)

    return {
        "failure_probability": round(float(failure_prob), 3),
        "shap_summary": shap_summary
    }