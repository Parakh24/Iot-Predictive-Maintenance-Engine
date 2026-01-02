from flask import Flask, request, jsonify
from inference import predict_failure

app = Flask(__name__)

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

        result = predict_failure(data)

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
    app.run(debug=True)
