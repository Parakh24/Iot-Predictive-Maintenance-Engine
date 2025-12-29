# SHAP Integration for XGBoost Model Explainability

## Overview

This module provides SHAP (SHapley Additive exPlanations) integration for explaining XGBoost model predictions in the IoT Predictive Maintenance Engine. SHAP values help understand which features contribute most to machine failure predictions.

## Features

- Compute SHAP values for XGBoost models
- Generate multiple visualization types:
  - Summary Plot (global feature impact with direction)
  - Bar Plot (global feature importance)
  - Force Plot (local explanation for single machine instance)
  - Decision Plot (feature contribution paths)
  - Waterfall Plot (detailed breakdown for single instance)
- Export feature importance rankings
- Save SHAP values for further analysis

## Installation

Required dependencies:

```bash
pip install shap matplotlib pandas numpy joblib xgboost scikit-learn
```

## Usage

### Quick Start

Run the complete SHAP analysis with default settings:

```bash
cd d:/IOT_Project
python -m src.explain.run_shap_analysis
```

### Using the SHAPExplainer Class

```python
from src.explain.shap_explainer import SHAPExplainer
import joblib
import pandas as pd

model = joblib.load("path/to/model.joblib")
X_data = pd.read_csv("path/to/data.csv")

explainer = SHAPExplainer(model=model, X_data=X_data)

explainer.compute_shap_values()

explainer.summary_plot(output_path="summary.png")
explainer.bar_plot(output_path="bar.png")
explainer.force_plot(instance_index=0, output_path="force.png")
explainer.decision_plot(output_path="decision.png")

importance = explainer.get_feature_importance()
print(importance.head(10))
```

### Custom Analysis

```python
from src.explain.run_shap_analysis import run_complete_shap_analysis

explainer, importance = run_complete_shap_analysis(
    data_path="data/processed/feature_engineered_data.csv",
    model_path="path/to/custom_model.joblib",
    scaler_path="path/to/scaler.joblib",
    output_dir="custom_output_dir",
    sample_size=1000,
    instance_index=5,
    max_display=15
)
```

## Output Files

After running the analysis, the following files are generated in the output directory:

| File | Description |
|------|-------------|
| `shap_summary_plot.png` | Beeswarm plot showing feature impact distribution |
| `shap_bar_plot.png` | Bar chart of global feature importance |
| `shap_force_plot_instance_X.png` | Force plot for specific machine instance |
| `shap_decision_plot.png` | Decision plot showing feature contribution paths |
| `shap_waterfall_plot_instance_X.png` | Waterfall breakdown for specific instance |
| `feature_importance.csv` | CSV file with ranked feature importance values |
| `shap_values.joblib` | Serialized SHAP values for further analysis |
| `shap_analysis_summary.txt` | Text summary of the analysis results |

## Interpreting Results

### Summary Plot
- Each dot represents a sample
- X-axis shows SHAP value (impact on prediction)
- Color indicates feature value (red=high, blue=low)
- Features are ranked by importance

### Bar Plot
- Shows mean absolute SHAP value for each feature
- Higher values indicate more important features

### Force Plot
- Red features push prediction toward failure
- Blue features push prediction away from failure
- Length indicates magnitude of contribution

### Decision Plot
- Shows cumulative feature contributions
- Each line represents one prediction path
- Helps identify which features differentiate outcomes

## Module Structure

```
src/explain/
    __init__.py
    shap_explainer.py
    run_shap_analysis.py
    README.md
    shap_outputs/
        shap_summary_plot.png
        shap_bar_plot.png
        shap_force_plot_instance_0.png
        shap_decision_plot.png
        shap_waterfall_plot_instance_0.png
        feature_importance.csv
        shap_values.joblib
        shap_analysis_summary.txt
```

## API Reference

### SHAPExplainer Class

#### Constructor
- `__init__(model, X_data, feature_names=None)`: Initialize explainer with model and data

#### Methods
- `compute_shap_values()`: Compute SHAP values using TreeExplainer
- `get_shap_values()`: Get computed SHAP values
- `summary_plot(output_path, max_display)`: Generate summary beeswarm plot
- `bar_plot(output_path, max_display)`: Generate bar plot of feature importance
- `force_plot(instance_index, output_path)`: Generate force plot for single instance
- `decision_plot(instance_indices, output_path, max_display)`: Generate decision plot
- `waterfall_plot(instance_index, output_path, max_display)`: Generate waterfall plot
- `get_feature_importance()`: Get DataFrame of feature importance rankings
- `save_shap_values(output_path)`: Save SHAP values to file
- `generate_all_plots(output_dir, instance_index, max_display)`: Generate all visualizations

### run_complete_shap_analysis Function

```python
run_complete_shap_analysis(
    data_path=None,
    model_path=None,
    scaler_path=None,
    output_dir=None,
    sample_size=500,
    instance_index=0,
    max_display=20
)
```

Returns: `(SHAPExplainer, DataFrame)` - The explainer instance and feature importance DataFrame
