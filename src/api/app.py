from flask import Flask, request, jsonify
import joblib
from inference import predict_failure
import shap 

app = Flask(__name__)

# Load model artifacts ONCE at startup
model = joblib.load("models/final_xgb_model.joblib")
preprocessor = joblib.load("models/preprocessing_pipeline.joblib")

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

        result = predict_failure(data, model, preprocessor)                            #before only data as arguments after, optimization model and preprocessor

        prob = result["failure_probability"]
        risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"

        return jsonify({
            "failure_probability": prob,
            "risk_level": risk,
            "shap_summary": result["shap_summary"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)                           #before True but after optimization false is there 


    #what i have done is big latency improvement work done 
