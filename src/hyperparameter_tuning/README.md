# 🔧 Hyperparameter Tuning Module

**Week 2 Deliverable: Modeling & Hyperparameter Tuning**

This module provides comprehensive hyperparameter optimization for the IoT Predictive Maintenance project, implementing the Week 2 requirements for hyperparameter tuning.

---

## 📋 Week 2 Requirements Covered

| Requirement | Implementation |
|-------------|----------------|
| ✅ RandomizedSearchCV for XGBoost | `hyperparameter_optimizer.py` |
| ✅ Time-series split validation | Using `TimeSeriesSplit` |
| ✅ Focus on Recall and F1 | Primary scoring metrics |
| ✅ n_estimators tuning | Range: 100-500 |
| ✅ max_depth tuning | Range: 3-10 |
| ✅ learning_rate tuning | Range: 0.01-0.3 |
| ✅ subsample tuning | Range: 0.5-1.0 |
| ✅ colsample_bytree tuning | Range: 0.5-1.0 |
| ✅ Class imbalance handling | Using `scale_pos_weight` and `class_weight='balanced'` |

---

## 📁 Module Structure

```
src/hyperparameter_tuning/
├── __init__.py                    # Module initialization
├── config.py                      # Configuration and search spaces
├── utils.py                       # Utility functions
├── hyperparameter_optimizer.py    # Main optimizer (RandomizedSearchCV)
├── optuna_optimizer.py            # Advanced optimizer (Bayesian)
├── run_optimization.py            # Main execution script
├── README.md                      # This documentation
├── results/                       # Optimization results (auto-created)
│   └── optuna/                    # Optuna-specific results
└── optimized_models/              # Saved models (auto-created)
```

---

## 🚀 Quick Start

### Option 1: Full Optimization (All Models)

```bash
cd d:/IOT_Project
python src/hyperparameter_tuning/run_optimization.py --mode full
```

### Option 2: Quick Test Run

```bash
python src/hyperparameter_tuning/run_optimization.py --mode quick
```

### Option 3: XGBoost Only (Week 2 Focus)

```bash
python src/hyperparameter_tuning/run_optimization.py --mode xgboost
```

### Option 4: Advanced Optuna Optimization

```bash
python src/hyperparameter_tuning/optuna_optimizer.py
```

---

## 📊 Models Optimized

### 1. Logistic Regression (Baseline)
- **Purpose**: Benchmark performance
- **Metrics**: Accuracy, Precision, Recall, F1

### 2. Random Forest Classifier
- **Purpose**: Ensemble method comparison
- **Parameters Tuned**: n_estimators, max_depth, min_samples_split, etc.

### 3. XGBoost Classifier (Primary Focus)
- **Purpose**: Main production model
- **Parameters Tuned** (as per Week 2):
  - `n_estimators`: [100, 200, 300, 400, 500]
  - `max_depth`: [3, 4, 5, 6, 7, 8, 10]
  - `learning_rate`: [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
  - `subsample`: [0.6, 0.7, 0.8, 0.9, 1.0]
  - `colsample_bytree`: [0.6, 0.7, 0.8, 0.9, 1.0]

---

## 🏆 Optimization Results

### Best XGBoost Parameters (RandomizedSearchCV)

| Parameter | Best Value |
|-----------|------------|
| n_estimators | 500 |
| max_depth | 8 |
| learning_rate | 0.15 |
| subsample | 0.8 |
| colsample_bytree | 0.7 |
| gamma | 0.2 |
| reg_alpha | 0.01 |
| reg_lambda | 2 |
| min_child_weight | 1 |

### Model Performance Comparison

| Model | CV Score | Accuracy | Precision | Recall | F1 Score |
|-------|----------|----------|-----------|--------|----------|
| **XGBoost** | 0.9550 | **0.9985** | 1.0000 | **0.9231** | **0.9600** |
| Random Forest | 0.9363 | 0.9860 | 1.0000 | 0.8780 | 0.9351 |
| Logistic Regression | 0.5870 | 0.9360 | 0.3036 | 0.8462 | 0.4468 |

### Key Findings

- **XGBoost achieved the best performance** with F1 Score of 0.96 and Recall of 0.92
- **Time-series validation** ensured realistic performance estimation
- **Class imbalance handling** using `scale_pos_weight` significantly improved recall
- **Deeper trees (max_depth=8)** with more estimators (500) provided best results

### Optuna Bayesian Optimization Results

| Model | CV Score | Test F1 | Test Recall | Trials |
|-------|----------|---------|-------------|--------|
| XGBoost | 0.9231 | 0.8503 | 0.8974 | 50 |
| Random Forest | 0.9231 | 0.8974 | 0.8974 | 50 |

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
OPTIMIZATION_CONFIG = {
    "n_cv_splits": 5,           # Cross-validation folds
    "n_iter": 50,               # RandomizedSearchCV iterations
    "primary_metric": "f1",     # Scoring metric
    "random_state": 42,         # For reproducibility
}
```

---

## 📈 Outputs

### Results Directory (`src/hyperparameter_tuning/results/`)
- `model_comparison.csv` - Side-by-side comparison of all models
- `xgboost_tuning_results_<timestamp>.json` - XGBoost best parameters
- `randomforest_tuning_results_<timestamp>.json` - RF best parameters
- `optimization_summary_<timestamp>.json` - Overall summary

### Models Directory (`src/hyperparameter_tuning/optimized_models/`)
- `xgboost_optimized_<timestamp>.joblib` - Best XGBoost model
- `randomforest_optimized_<timestamp>.joblib` - Best RF model
- `scaler_<timestamp>.joblib` - Fitted StandardScaler
- `feature_names_<timestamp>.json` - Feature column names

---

## 🔬 Optimization Methods

### RandomizedSearchCV (Standard)
- Randomly samples from parameter distributions
- Fast and effective for large search spaces
- Uses TimeSeriesSplit for proper validation

### Optuna (Advanced)
- Bayesian optimization (TPE Sampler)
- Intelligent exploration of parameter space
- Early pruning of unpromising trials
- Better for finding optimal parameters

---

## 📝 Usage Examples

### Programmatic Usage

```python
from src.hyperparameter_tuning import HyperparameterOptimizer

# Initialize
optimizer = HyperparameterOptimizer(
    n_cv_splits=5,
    n_iter=50,
    scoring='f1'
)

# Run optimization
results = optimizer.run_full_optimization()

# Get best XGBoost parameters
best_params = optimizer.get_best_params('XGBoost')
print(best_params)

# Get best model
best_xgb = optimizer.get_best_model('XGBoost')
```

### Custom Parameter Grid

```python
from src.hyperparameter_tuning import HyperparameterOptimizer

optimizer = HyperparameterOptimizer()
optimizer.load_and_prepare_data()

# Custom XGBoost grid
custom_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}

result = optimizer.optimize_xgboost(param_grid=custom_grid, n_iter=30)
```

---

## 📦 Dependencies

```
scikit-learn>=1.0.0
xgboost>=1.5.0
pandas>=1.3.0
numpy>=1.20.0
joblib>=1.0.0
imbalanced-learn>=0.9.0  # For SMOTE (optional)
optuna>=3.0.0            # For advanced optimization (optional)
```

Install with:
```bash
pip install scikit-learn xgboost pandas numpy joblib imbalanced-learn optuna
```

---

## 🎯 Week 2 Deliverable Checklist

- [x] Hyperparameter optimization module created
- [x] RandomizedSearchCV implemented for XGBoost
- [x] All required parameters tuned (n_estimators, max_depth, learning_rate, subsample, colsample_bytree)
- [x] Time-series split validation
- [x] Focus on Recall and F1 metrics
- [x] Class imbalance handling
- [x] Model comparison report generation
- [x] Saved candidate models
- [x] Documentation complete

---

## 👥 Team Integration

This module is designed to work seamlessly with other Week 2 components:
- Uses data from `data/processed/feature_engineered_data.csv`
- Does NOT modify any existing files
- Creates its own output directories
- Models can be easily integrated into `model_train.py`

---

## 📧 Contact

For questions about this module, contact the Hyperparameter Optimization team member.
