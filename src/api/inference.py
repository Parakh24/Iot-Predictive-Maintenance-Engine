import time
import pandas as pd
import numpy as np
import shap

#optimized inference by removing repeated model loading, switching from kernel SHAP to TreeExplainer, 
#and eliminating unnecessary high-dimensional feature construction.

def predict_failure(sensor_json: dict, model, preprocessor):
    start_time = time.perf_counter()

    # Clean input
    clean_data = {k: float(v) for k, v in sensor_json.items()}
    input_df = pd.DataFrame([clean_data])

    # Preprocess (handles feature expansion internally)
    X = preprocessor.transform(input_df)

    # Predict probability
    failure_prob = model.predict_proba(X)[0, 1]

    # FAST SHAP for XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Top 4 features only (since input has 4)
    shap_summary = {
        feature: round(float(abs(shap_values[0][i])), 4)
        for i, feature in enumerate(input_df.columns)
    }

    latency_ms = (time.perf_counter() - start_time) * 1000
    print(f"Inference latency: {latency_ms:.2f} ms")

    return {
        "failure_probability": round(float(failure_prob), 3),
        "shap_summary": shap_summary,
        "latency_ms": latency_ms
    }
