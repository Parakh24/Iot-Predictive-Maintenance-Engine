import time
import pandas as pd
import numpy as np
import shap

#optimized inference by removing repeated model loading, switching from kernel SHAP to TreeExplainer, 
#and eliminating unnecessary high-dimensional feature construction.

def predict_failure(sensor_json: dict, pipeline):
    """
    Predict failure using the full pipeline (preprocessor + model).
    
    Args:
        sensor_json: Dictionary with all sensor readings and engineered features
        pipeline: Full sklearn Pipeline with preprocessor and model steps
    
    Returns:
        Dictionary with failure_probability, shap_summary, and latency_ms
    """
    start_time = time.perf_counter()

    # Create DataFrame from input
    input_df = pd.DataFrame([sensor_json])

    # Use the full pipeline to predict (it handles preprocessing internally)
    failure_prob = pipeline.predict_proba(input_df)[0, 1]

    # FAST SHAP for XGBoost - extract the model step from pipeline
    model_step = pipeline.named_steps['model']
    
    # Get preprocessed data for SHAP
    preprocessor_step = pipeline.named_steps['preprocessor']
    X_processed = preprocessor_step.transform(input_df)
    
    explainer = shap.TreeExplainer(model_step)
    shap_values = explainer.shap_values(X_processed)

    # Get top 5 most important features for this prediction
    if len(shap_values.shape) == 1:
        sv = shap_values
    else:
        sv = shap_values[0] if shap_values.ndim == 2 else shap_values
    
    # Get feature importance with absolute values
    feature_importance = [(i, abs(float(sv[i]))) for i in range(len(sv))]
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 5 features
    top_features = feature_importance[:5]
    shap_summary = {f"feature_{i+1}": round(imp, 4) for i, (_, imp) in enumerate(top_features)}

    latency_ms = (time.perf_counter() - start_time) * 1000
    print(f"Inference latency: {latency_ms:.2f} ms")

    return {
        "failure_probability": round(float(failure_prob), 3),
        "shap_summary": shap_summary,
        "latency_ms": latency_ms
    }
