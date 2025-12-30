import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# 1. Load dataset
# -------------------------------------------------
df = pd.read_csv("data/processed/feature_engineered_data.csv")

# -------------------------------------------------
# 2. Define target and drop non-feature columns
# -------------------------------------------------
TARGET_COL = "Machine failure"

DROP_COLS = [
    "UDI",
    "Product ID",
    "Type",
    "Machine failure",
    "TWF", "HDF", "PWF", "OSF", "RNF"
]

y = df[TARGET_COL]
X = df.drop(columns=DROP_COLS)

# -------------------------------------------------
# 3. Train model
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# -------------------------------------------------
# 4. Compute SHAP values
# -------------------------------------------------
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test, check_additivity=False).values

# -------------------------------------------------
# 5. Domain validation (manufacturing logic)
# -------------------------------------------------
domain_expectations = {
    "Air temperature [K]": "positive",
    "Process temperature [K]": "positive",
    "Torque [Nm]": "positive",
    "Tool wear [min]": "positive"
}

print("\nDOMAIN VALIDATION REPORT")
print("-" * 55)

for feature, expected in domain_expectations.items():
    if feature not in X_test.columns:
        print(f"{feature:30s} | Feature not found")
        continue

    idx = X_test.columns.get_loc(feature)
    # Use failure-class SHAP values
    corr, _ = spearmanr(X_test[feature], shap_values[:, idx, 1])

    status = "OK"
    if expected == "positive" and corr < 0:
        status = "Violates domain logic"
    if expected == "negative" and corr > 0:
        status = "Violates domain logic"

    print(f"{feature:30s} | SHAP Corr: {corr: .3f} | {status}")

# -------------------------------------------------
# 6. Anomaly detection (root-cause analysis)
# -------------------------------------------------
anomalies = []

for i in range(len(X_test)):
    row = X_test.iloc[i]
    # Use failure-class SHAP values
    shap_row = shap_values[i, :, 1]

    if row["Air temperature [K]"] < X_test["Air temperature [K]"].median() and \
       shap_row[X_test.columns.get_loc("Air temperature [K]")] > 0:
        anomalies.append((i, "High risk at low air temperature"))

    if row["Torque [Nm]"] < X_test["Torque [Nm]"].median() and \
       shap_row[X_test.columns.get_loc("Torque [Nm]")] > 0:
        anomalies.append((i, "High risk at low torque"))

    if row["Tool wear [min]"] < X_test["Tool wear [min]"].median() and \
       shap_row[X_test.columns.get_loc("Tool wear [min]")] > 0:
        anomalies.append((i, "High risk at low tool wear"))

anomaly_df = pd.DataFrame(anomalies, columns=["Row_Index", "Issue"])

print("\nANOMALIES REQUIRING ROOT-CAUSE ANALYSIS")
print("-" * 55)
print(anomaly_df if not anomaly_df.empty else "None detected")
