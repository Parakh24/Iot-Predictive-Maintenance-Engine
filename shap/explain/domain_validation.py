import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

# -------------------------------
# Inputs
# -------------------------------
# model : trained ML model
# X     : pandas DataFrame with columns:
#        ['Temperature', 'Vibration', 'Pressure_stability']

# -------------------------------
# Compute SHAP values
# -------------------------------
explainer = shap.Explainer(model, X)
shap_values = explainer(X).values

# -------------------------------
# Domain validation logic
# -------------------------------
domain_expectations = {
    "Temperature": "positive",
    "Vibration": "positive",
    "Pressure_stability": "negative"
}

print("\nDOMAIN VALIDATION REPORT")
print("-" * 40)

for feature, expected in domain_expectations.items():
    corr, _ = spearmanr(X[feature], shap_values[:, X.columns.get_loc(feature)])

    status = "OK"
    if expected == "positive" and corr < 0:
        status = "Violates domain logic"
    if expected == "negative" and corr > 0:
        status = "Violates domain logic"

    print(f"{feature:20s} | SHAP Corr: {corr: .3f} | {status}")

# -------------------------------
# Anomaly detection
# -------------------------------
anomalies = []

for i in range(len(X)):
    if X.iloc[i]["Temperature"] < X["Temperature"].median() and \
       shap_values[i, X.columns.get_loc("Temperature")] > 0:
        anomalies.append((i, "High risk at low temperature"))

    if X.iloc[i]["Vibration"] < X["Vibration"].median() and \
       shap_values[i, X.columns.get_loc("Vibration")] > 0:
        anomalies.append((i, "High risk at low vibration"))

    if X.iloc[i]["Pressure_stability"] > X["Pressure_stability"].median() and \
       shap_values[i, X.columns.get_loc("Pressure_stability")] > 0:
        anomalies.append((i, "High risk despite stable pressure"))

anomaly_df = pd.DataFrame(anomalies, columns=["Row", "Issue"])

print("\nANOMALIES REQUIRING ROOT-CAUSE ANALYSIS")
print("-" * 40)
print(anomaly_df if not anomaly_df.empty else "None detected")
