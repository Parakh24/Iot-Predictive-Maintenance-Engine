import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
import glob
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from explain.shap_explainer import SHAPExplainer


def load_and_prepare_data(data_path, target_column="Machine failure"):
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    non_numeric_cols = X.select_dtypes(include=["object"]).columns
    X = X.drop(columns=non_numeric_cols)
    
    X.columns = [str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_').replace(' ', '_') for col in X.columns]
    
    return X, y


def run_complete_shap_analysis(
    data_path=None,
    output_dir=None,
    sample_size=500,
    instance_index=0,
    max_display=20
):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if data_path is None:
        data_path = os.path.join(base_dir, "data", "processed", "feature_engineered_data.csv")
    
    if output_dir is None:
        output_dir = os.path.join(base_dir, "src", "explain", "shap_outputs")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("SHAP ANALYSIS FOR XGBOOST MODEL")
    print("=" * 60)
    print(f"\nData path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Sample size: {sample_size}")
    print(f"Instance index for local explanation: {instance_index}")
    print("=" * 60)
    
    print("\nLoading data...")
    X, y = load_and_prepare_data(data_path)
    feature_names = X.columns.tolist()
    print(f"Data shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    print("\nScaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining XGBoost model...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    print("Model trained successfully")
    
    joblib.dump(model, os.path.join(output_dir, 'xgboost_model.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)
    
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    if sample_size and sample_size < len(X_test_scaled_df):
        np.random.seed(42)
        sample_indices = np.random.choice(len(X_test_scaled_df), sample_size, replace=False)
        X_sample = X_test_scaled_df.iloc[sample_indices].reset_index(drop=True)
        print(f"Sampled {sample_size} instances for SHAP analysis")
    else:
        X_sample = X_test_scaled_df.reset_index(drop=True)
        print(f"Using all {len(X_sample)} test instances for SHAP analysis")
    
    print("\nInitializing SHAP Explainer...")
    explainer = SHAPExplainer(
        model=model,
        X_data=X_sample,
        feature_names=feature_names
    )
    
    print("\nComputing SHAP values...")
    explainer.compute_shap_values()
    print("SHAP values computed successfully")
    
    print("\nGenerating all SHAP visualizations...")
    importance_df = explainer.generate_all_plots(
        output_dir=output_dir,
        instance_index=instance_index,
        max_display=max_display
    )
    
    summary_path = os.path.join(output_dir, "shap_analysis_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("SHAP ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Sample size: {len(X_sample)}\n")
        f.write(f"Number of features: {len(feature_names)}\n\n")
        f.write("TOP 10 MOST IMPORTANT FEATURES:\n")
        f.write("-" * 40 + "\n")
        for idx, row in importance_df.head(10).iterrows():
            f.write(f"{idx+1}. {row['feature']}: {row['importance']:.6f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Generated Outputs:\n")
        f.write("- shap_summary_plot.png\n")
        f.write("- shap_bar_plot.png\n")
        f.write(f"- shap_force_plot_instance_{instance_index}.png\n")
        f.write("- shap_decision_plot.png\n")
        f.write(f"- shap_waterfall_plot_instance_{instance_index}.png\n")
        f.write("- feature_importance.csv\n")
        f.write("- shap_values.joblib\n")
        f.write("- xgboost_model.joblib\n")
        f.write("- scaler.joblib\n")
        f.write("- feature_names.json\n")
    
    print("\n" + "=" * 60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 60)
    print(importance_df.head(10).to_string(index=False))
    
    print(f"\nSummary saved to: {summary_path}")
    print("\n" + "=" * 60)
    print("SHAP ANALYSIS COMPLETED SUCCESSFULLY")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)
    
    return explainer, importance_df


if __name__ == "__main__":
    run_complete_shap_analysis()
