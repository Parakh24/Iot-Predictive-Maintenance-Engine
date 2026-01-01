import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
print(f"Models will be saved in: {MODELS_DIR}")

# ------------------------------
# Find train.csv dynamically
# ------------------------------
train_file_found = False
for root, dirs, files in os.walk(BASE_DIR):
    if "train.csv" in files:
        DATA_PATH = os.path.join(root, "train.csv")
        train_file_found = True
        break

if not train_file_found:
    raise FileNotFoundError(f"train.csv not found anywhere under {BASE_DIR}")

print(f"Using train.csv at: {DATA_PATH}")

# ------------------------------
# Load data
# ------------------------------
train = pd.read_csv(DATA_PATH)
if "Machine failure" not in train.columns:
    raise KeyError("'Machine failure' column not found in the dataset.")

X_train = train.drop(columns=["Machine failure"])
y_train = train["Machine failure"]

# ------------------------------
# Identify categorical and numerical columns
# ------------------------------
categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# ------------------------------
# Create preprocessor
# ------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ]
)

# ------------------------------
# Create and train pipelines
# ------------------------------
pipelines = {
    "Baseline": LogisticRegression(class_weight="balanced", max_iter=1000),
    "ImbalanceHandled": LogisticRegression(class_weight="balanced", max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

for name, model in pipelines.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)
    save_path = os.path.join(MODELS_DIR, f"{name.lower()}_pipeline.joblib")
    joblib.dump(pipeline, save_path)
    print(f"{name} pipeline trained and saved at: {save_path}")

print("All pipelines trained and saved successfully!")
