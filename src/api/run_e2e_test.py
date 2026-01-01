import subprocess
import time
import requests
import json
import sys
import os
import signal

def run_test():
    print("="*60)
    print("STARTING END-TO-END TEST FOR WEEK 4: DEPLOYMENT WRAPPER")
    print("="*60)

    # 1. Start the API Server
    print("\n[Step 1] Starting API Server...")
    # Using python -m uvicorn ... to ensure path is correct
    # We assume we are in d:\IOT_Project root
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.app:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd()
    )
    
    # Wait for server to start
    print("Waiting for server to initialize (10 seconds)...")
    time.sleep(10)
    
    # Check if process is still running
    if api_process.poll() is not None:
        stdout, stderr = api_process.communicate()
        print("API Server failed to start!")
        print("STDOUT:", stdout.decode())
        print("STDERR:", stderr.decode())
        return

    try:
        # 2. Simulate Incoming Sensor Reading
        print("\n[Step 2] Simulating Incoming Sensor Reading...")
        # Sample data taken from feature engineered dataset
        # This represents a single instance of machine state
        sample_payload = {
            "UDI": 1,
            "Product ID": "M14860", # API should ignore or handle this
            "Type": "M", # API should ignore or handle this if features are present
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1551,
            "Torque [Nm]": 42.8,
            "Tool wear [min]": 0,
            # "Machine failure": 0, # Target not sent in inference
            "TWF": 0,
            "HDF": 0,
            "PWF": 0,
            "OSF": 0,
            "RNF": 0,
            "Temperature_difference [K]": 10.5,
            "Power [W]": 6951.59056,
            "Wear_Torque_Interaction": 0.0,
            "Air temperature [K]_rolling_mean_3": 298.1,
            "Air temperature [K]_rolling_std_3": 0.0,
            "Process temperature [K]_rolling_mean_3": 308.6,
            "Process temperature [K]_rolling_std_3": 0.0,
            "Rotational speed [rpm]_rolling_mean_3": 1551.0,
            "Rotational speed [rpm]_rolling_std_3": 0.0,
            "Torque [Nm]_rolling_mean_3": 42.8,
            "Torque [Nm]_rolling_std_3": 0.0,
            "Air temperature [K]_rolling_mean_5": 298.1,
            "Air temperature [K]_rolling_std_5": 0.0,
            "Process temperature [K]_rolling_mean_5": 308.6,
            "Process temperature [K]_rolling_std_5": 0.0,
            "Rotational speed [rpm]_rolling_mean_5": 1551.0,
            "Rotational speed [rpm]_rolling_std_5": 0.0,
            "Torque [Nm]_rolling_mean_5": 42.8,
            "Torque [Nm]_rolling_std_5": 0.0,
            "Type_Ordinal": 2
        }
        
        print("Sample Data prepared:")
        print(json.dumps(sample_payload, indent=2))

        # 3. Call API and Confirm Correct JSON Output
        print("\n[Step 3] Sending Request to API (POST /predict)...")
        url = "http://127.0.0.1:8000/predict"
        
        try:
            response = requests.post(url, json=sample_payload)
            response.raise_for_status()
            result = response.json()
            
            print("\n[Step 4] Validating Response...")
            print("Response Status Code:", response.status_code)
            print("Raw Response JSON:")
            print(json.dumps(result, indent=2))
            
            # Validation logic
            required_keys = ["prediction", "probability", "status", "inference_time_ms"]
            missing_keys = [k for k in required_keys if k not in result]
            
            if missing_keys:
                print(f"FAILED: Missing keys in response: {missing_keys}")
            else:
                print("SUCCESS: JSON output structure is correct.")
                
                # Check specifics
                pred = result["prediction"]
                prob = result["probability"]
                status = result["status"]
                
                print(f"Prediction: {pred} (0=Normal, 1=Failure)")
                print(f"Probability: {prob:.4f}")
                print(f"Status Message: {status}")
                
                if isinstance(pred, int) and isinstance(prob, float) and isinstance(status, str):
                    print("Data types validated.")
                else:
                    print("FAILED: Data types improper.")

        except requests.exceptions.RequestException as e:
            print(f"FAILED: Request Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print("Server Response:", e.response.text)
                
        # 4. Log Inference Time
        print("\n[Step 5] Logging Inference Time...")
        if 'result' in locals() and 'inference_time_ms' in result:
            inf_time = result['inference_time_ms']
            print(f"Server-side Inference Time: {inf_time} ms")
            
            # Simple threshold check (just for info)
            if inf_time < 100:
                print("Performance: Excellent (<100ms)")
            else:
                print("Performance: Acceptable")
        else:
            print("Could not retrieve inference time.")

    finally:
        # Cleanup
        print("\n[Step 6] Shutting down API Server...")
        api_process.terminate()
        try:
            api_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            api_process.kill()
        print("Test Completed.")
        print("="*60)

if __name__ == "__main__":
    run_test()
