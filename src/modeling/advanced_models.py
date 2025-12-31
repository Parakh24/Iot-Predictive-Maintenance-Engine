import pandas as pd
import numpy as np
import os
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score
from xgboost import XGBClassifier

# Load data
DATA_PATH = "data/processed/feature_engineered_data.csv"
df = pd.read_csv(DATA_PATH)

# Target and features
TARGET = "Machine failure"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Drop non-numeric columns
non_numeric_cols = X.select_dtypes(include=["object"]).columns
print("Dropping non-numeric columns:", list(non_numeric_cols))
X = X.drop(columns=non_numeric_cols)

# Clean column names for XGBoost (no special chars)
X.columns = [str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') for col in X.columns]

# Time-based split
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)

print("\n--- Random Forest ---")
print("Recall:", recall_score(y_test, rf_preds))
print("F1:", f1_score(y_test, rf_preds))

# XGBoost (Main Model)
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train_scaled, y_train)
xgb_preds = xgb.predict(X_test_scaled)

print("\n--- XGBoost ---")
print("Recall:", recall_score(y_test, xgb_preds))
print("F1:", f1_score(y_test, xgb_preds))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(xgb, "src/modeling/models/xgboost_model.joblib")
joblib.dump(scaler, "src/modeling/models/xgboost_scaler.joblib")

print("\n Advanced models trained and saved successfully.")

