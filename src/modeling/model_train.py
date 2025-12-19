import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ------------------------------
# Paths
# ------------------------------

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"BASE_DIR: {BASE_DIR}")

# Models directory (absolute path)
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
print("Data loaded successfully!")

# ------------------------------
# Separate features and target
# ------------------------------
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
# Create pipelines
# ------------------------------
# Baseline model pipeline
baseline_model = LogisticRegression(class_weight="balanced", max_iter=1000)
baseline_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", baseline_model)
])

# Fit baseline pipeline
baseline_pipeline.fit(X_train, y_train)
print("Baseline model trained successfully.")

# Save baseline pipeline
baseline_pipeline_path = os.path.join(MODELS_DIR, "baseline_pipeline.joblib")
joblib.dump(baseline_pipeline, baseline_pipeline_path)
print(f"Baseline pipeline saved at: {baseline_pipeline_path}")

# ------------------------------
# Imbalance-handled model pipeline (optional: here we use same as baseline)
# ------------------------------
# You can modify class_weight or sampling strategy for imbalance-handled model
imbalance_model = LogisticRegression(class_weight="balanced", max_iter=1000)
imbalance_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", imbalance_model)
])

imbalance_pipeline.fit(X_train, y_train)
imbalance_pipeline_path = os.path.join(MODELS_DIR, "imbalance_pipeline.joblib")
joblib.dump(imbalance_pipeline, imbalance_pipeline_path)
print(f"Imbalance-handled pipeline saved at: {imbalance_pipeline_path}")

print("All done!")
