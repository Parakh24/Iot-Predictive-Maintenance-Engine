import os
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "week2")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------
# Load test data
# -------------------------------
if not os.path.exists(TEST_CSV):
    raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")

test = pd.read_csv(TEST_CSV)
if "Machine failure" not in test.columns:
    raise ValueError("Test CSV must contain 'Machine failure' column")

X_test = test.drop("Machine failure", axis=1)
y_test = test["Machine failure"]

# -------------------------------
# Define pipeline paths
# -------------------------------
pipelines = {
    "Baseline": os.path.join(BASE_DIR, "models", "baseline_pipeline.joblib"),
    "ImbalanceHandled": os.path.join(BASE_DIR, "models", "imbalance_pipeline.joblib"),
    "RandomForest": os.path.join(BASE_DIR, "models", "randomforest_pipeline.joblib"),
    "XGBoost": os.path.join(BASE_DIR, "models", "xgboost_pipeline.joblib")
}

results = []

# -------------------------------
# Evaluate pipelines
# -------------------------------
for name, pipeline_path in pipelines.items():
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

    # Load pipeline (preprocessing + model)
    pipeline = joblib.load(pipeline_path)

    # Predict directly; pipeline handles preprocessing
    y_pred = pipeline.predict(X_test)

    # Metrics
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"\n=== {name} ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    results.append({
        "Model": name,
        "F1": f1,
        "Recall": recall
    })

# -------------------------------
# Save comparison report
# -------------------------------
output_csv = os.path.join(RESULTS_DIR, "model_comparison.csv")
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"\nModel comparison saved to: {output_csv}")
