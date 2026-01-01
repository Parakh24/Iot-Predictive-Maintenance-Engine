# Week 4: Deployment Wrapper & End-to-End Test

This module implements the Deployment Wrapper (API) and the End-to-End Test for the IoT Predictive Maintenance Engine.

## Contents
- `app.py`: FastAPI application serving the machine learning model.
- `run_e2e_test.py`: End-to-End test script that simulates sensor data and validates the API.

## Setup

1. Install dependencies:
   ```bash
   pip install fastapi uvicorn requests pydantic
   ```

2. Ensure model artifacts exist in `src/explain/shap_outputs/` (generated from Week 3).

## Running the End-to-End Test

To run variables automated test (including server startup/shutdown):

```bash
python src/api/run_e2e_test.py
```

## Running the API Manually

1. Start the server:
   ```bash
   python src/api/app.py
   ```
   OR
   ```bash
   uvicorn src.api.app:app --host 0.0.0.0 --port 8000
   ```

2. Access Swagger documentation at: `http://localhost:8000/docs`

## End-to-End Test Details

The test script performs the following validation steps:
1. **Starts the API Server** in a background process.
2. **Simulates an incoming sensor reading** (using a sample vector from feature engineering).
3. **Sends a POST request** to the `/predict` endpoint.
4. **Validates the JSON output** schema (Prediction, Probability, Status).
5. **Logs and checks inference time** (Performance verification).
6. **Cleans up** by shutting down the server.
