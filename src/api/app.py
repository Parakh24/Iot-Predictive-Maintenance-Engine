from flask import Flask, request, jsonify
import joblib
from inference import predict_failure
import os

app = Flask(__name__)

# -------------------------------
# Paths: load models from src/modeling/models/
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# XGBoost full pipeline (includes preprocessor + model)
pipeline_path = os.path.join(BASE_DIR, "../modeling/models/xgboost_pipeline.joblib")

# Load pipeline ONCE at startup (latency optimization)
pipeline = joblib.load(pipeline_path)

# Required input fields (matching feature engineered data)
REQUIRED_FIELDS = [
    "UDI", "Product ID", "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "TWF", "HDF", 
    "PWF", "OSF", "RNF", "Temperature_difference [K]", "Power [W]", 
    "Wear_Torque_Interaction", "Air temperature [K]_rolling_mean_3",
    "Air temperature [K]_rolling_std_3", "Process temperature [K]_rolling_mean_3",
    "Process temperature [K]_rolling_std_3", "Rotational speed [rpm]_rolling_mean_3",
    "Rotational speed [rpm]_rolling_std_3", "Torque [Nm]_rolling_mean_3",
    "Torque [Nm]_rolling_std_3", "Air temperature [K]_rolling_mean_5",
    "Air temperature [K]_rolling_std_5", "Process temperature [K]_rolling_mean_5",
    "Process temperature [K]_rolling_std_5", "Rotational speed [rpm]_rolling_mean_5",
    "Rotational speed [rpm]_rolling_std_5", "Torque [Nm]_rolling_mean_5",
    "Torque [Nm]_rolling_std_5", "Type_Ordinal"
]

# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    return {"status": "IoT Predictive Maintenance API running"}

# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Basic input validation
        missing_fields = [f for f in REQUIRED_FIELDS if f not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {missing_fields[:5]}..."}), 400

        # Call inference with full pipeline
        result = predict_failure(data, pipeline)

        prob = result["failure_probability"]
        risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"

        return jsonify({
            "failure_probability": prob,
            "risk_level": risk,
            "shap_summary": result["shap_summary"]
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    # debug=False for latency / production
    app.run(debug=False, host="0.0.0.0", port=5000)
