import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, f1_score

from imblearn.over_sampling import SMOTE

def load_data():
    return pd.read_csv("data/processed/cleaned_sensor_data.csv")

def prepare_features(df):
    X = df[
        [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
    ]
    y = df["Machine failure"]
    return X, y

def main():
    os.makedirs("models", exist_ok=True)

    df = load_data()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------------
    # 1️⃣ Baseline with class weights
    # -------------------------------
    weighted_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    weighted_model.fit(X_train, y_train)
    y_pred_weighted = weighted_model.predict(X_test)

    recall_weighted = recall_score(y_test, y_pred_weighted)
    f1_weighted = f1_score(y_test, y_pred_weighted)

    # -------------------------------
    # 2️⃣ SMOTE-based model
    # -------------------------------
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)

    smote_model = LogisticRegression(max_iter=1000)
    smote_model.fit(X_smote, y_smote)
    y_pred_smote = smote_model.predict(X_test)

    recall_smote = recall_score(y_test, y_pred_smote)
    f1_smote = f1_score(y_test, y_pred_smote)

    # -------------------------------
    # Results
    # -------------------------------
    print("Imbalance Handling Results")
    print("--------------------------")
    print(f"Class Weight - Recall : {recall_weighted:.4f}")
    print(f"Class Weight - F1     : {f1_weighted:.4f}")
    print()
    print(f"SMOTE - Recall        : {recall_smote:.4f}")
    print(f"SMOTE - F1            : {f1_smote:.4f}")

    # Save best-performing model (SMOTE usually better)
    joblib.dump(smote_model, "models/imbalance_handled_model.joblib")
    joblib.dump(scaler, "models/imbalance_scaler.joblib")

if __name__ == "__main__":
    main()
