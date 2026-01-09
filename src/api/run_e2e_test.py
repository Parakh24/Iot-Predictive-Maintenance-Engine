"""
Week 4 - Step 4: End-to-End Test for IoT Predictive Maintenance API
=====================================================================
This script performs automated end-to-end testing of the Flask REST API.

Tasks:
  1. Simulate incoming sensor reading
  2. Confirm correct JSON output
  3. Log inference time

Author: Week 4 - End-to-End Test Role
"""

import subprocess
import time
import requests
import json
import sys
import os


def run_e2e_test():
    """
    Execute the complete end-to-end test for the Predictive Maintenance API.
    """
    print("=" * 70)
    print("WEEK 4 - END-TO-END TEST: IoT Predictive Maintenance API")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Start the Flask API Server
    # =========================================================================
    print("\n[STEP 1] Starting Flask API Server...")
    
    api_script_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    # Start the server as a background process
    api_process = subprocess.Popen(
        [sys.executable, api_script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(__file__)
    )
    
    # Wait for server to initialize
    print("Waiting for server to initialize (10 seconds)...")
    time.sleep(10)
    
    # Check if server started successfully
    if api_process.poll() is not None:
        stdout, stderr = api_process.communicate()
        print("ERROR: API Server failed to start!")
        print("STDOUT:", stdout.decode())
        print("STDERR:", stderr.decode())
        return False

    print("Server started successfully on http://127.0.0.1:5000")

    try:
        # =====================================================================
        # STEP 2: Simulate Incoming Sensor Reading
        # =====================================================================
        print("\n[STEP 2] Simulating Incoming Sensor Reading...")
        
        # Sample sensor data simulating a machine's current state
        # Using the 4 fields expected by the API
        sensor_payload = {
            "temperature": 75.5,    # Temperature in Celsius
            "vibration": 0.45,      # Vibration intensity
            "pressure": 101.3,      # Pressure in kPa
            "humidity": 45.2        # Humidity percentage
        }
        
        print("Sensor Data Payload:")
        print(json.dumps(sensor_payload, indent=2))

        # =====================================================================
        # STEP 3: Send Request and Measure Response Time
        # =====================================================================
        print("\n[STEP 3] Sending POST Request to /predict Endpoint...")
        
        url = "http://127.0.0.1:5000/predict"
        
        # Measure client-side latency
        start_time = time.perf_counter()
        response = requests.post(url, json=sensor_payload, timeout=30)
        end_time = time.perf_counter()
        
        client_latency_ms = (end_time - start_time) * 1000
        
        print(f"HTTP Status Code: {response.status_code}")

        # =====================================================================
        # STEP 4: Validate JSON Output
        # =====================================================================
        print("\n[STEP 4] Validating JSON Output...")
        
        if response.status_code != 200:
            print(f"FAILED: Expected status 200, got {response.status_code}")
            print("Response:", response.text)
            return False
        
        result = response.json()
        print("Response JSON:")
        print(json.dumps(result, indent=2))
        
        # Validate required fields
        required_fields = ["failure_probability", "risk_level", "shap_summary"]
        missing_fields = [f for f in required_fields if f not in result]
        
        if missing_fields:
            print(f"FAILED: Missing required fields: {missing_fields}")
            return False
        
        print("\n--- Validation Results ---")
        print(f"✓ failure_probability: {result['failure_probability']} (type: {type(result['failure_probability']).__name__})")
        print(f"✓ risk_level: {result['risk_level']} (type: {type(result['risk_level']).__name__})")
        print(f"✓ shap_summary: {result['shap_summary']} (type: {type(result['shap_summary']).__name__})")
        
        # Type validation
        if not isinstance(result['failure_probability'], (int, float)):
            print("FAILED: failure_probability should be numeric")
            return False
        if not isinstance(result['risk_level'], str):
            print("FAILED: risk_level should be string")
            return False
        if not isinstance(result['shap_summary'], dict):
            print("FAILED: shap_summary should be a dictionary")
            return False
        
        print("\nJSON Structure Validation: PASSED")

        # =====================================================================
        # STEP 5: Log Inference Time
        # =====================================================================
        print("\n[STEP 5] Logging Inference Time...")
        
        print(f"Client-Side Round-Trip Latency: {client_latency_ms:.2f} ms")
        
        # Check target latency (< 50ms as per requirements)
        target_latency = 50.0
        if client_latency_ms < target_latency:
            print(f"Performance: EXCELLENT (< {target_latency}ms target)")
        elif client_latency_ms < 100:
            print(f"Performance: ACCEPTABLE (< 100ms)")
        else:
            print(f"Performance: NEEDS OPTIMIZATION (> 100ms)")

        # =====================================================================
        # SUMMARY
        # =====================================================================
        print("\n" + "=" * 70)
        print("END-TO-END TEST SUMMARY")
        print("=" * 70)
        print(f"  Endpoint Tested:      POST /predict")
        print(f"  HTTP Status:          {response.status_code} OK")
        print(f"  Failure Probability:  {result['failure_probability']}")
        print(f"  Risk Level:           {result['risk_level']}")
        print(f"  Latency:              {client_latency_ms:.2f} ms")
        print(f"  JSON Validation:      PASSED")
        print("=" * 70)
        print("TEST RESULT: ALL CHECKS PASSED")
        print("=" * 70)
        
        return True

    except requests.exceptions.ConnectionError:
        print("FAILED: Could not connect to API server")
        print("Ensure the Flask server is running on port 5000")
        return False
    except requests.exceptions.Timeout:
        print("FAILED: Request timed out")
        return False
    except Exception as e:
        print(f"FAILED: Unexpected error - {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # =====================================================================
        # CLEANUP: Shutdown the API Server
        # =====================================================================
        print("\n[CLEANUP] Shutting down API Server...")
        api_process.terminate()
        try:
            api_process.wait(timeout=5)
            print("Server shut down successfully.")
        except subprocess.TimeoutExpired:
            api_process.kill()
            print("Server force-killed.")


if __name__ == "__main__":
    success = run_e2e_test()
    sys.exit(0 if success else 1)
