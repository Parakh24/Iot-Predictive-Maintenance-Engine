from flask import Flask, request, jsonify
import joblib
from inference import predict_failure
import shap
import os

app = Flask(__name__)

# -------------------------------
# Paths: load models from src/modeling/models/
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# XGBoost model
model_path = os.path.join(BASE_DIR, "../modeling/models/xgboost_pipeline.joblib")
# Preprocessing pipeline (choose the correct pipeline)
preprocessor_path = os.path.join(BASE_DIR, "../modeling/models/imbalancehandled_pipeline.joblib")

# Load model artifacts ONCE at startup (latency optimization)
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

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
        required_fields = ["temperature", "vibration", "pressure", "humidity"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Call inference with model and preprocessor
        result = predict_failure(data, model, preprocessor)

        prob = result["failure_probability"]
        risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"

        return jsonify({
            "failure_probability": prob,
            "risk_level": risk,
            "shap_summary": result["shap_summary"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    # debug=False for latency / production
    app.run(debug=False, host="0.0.0.0", port=5000)
