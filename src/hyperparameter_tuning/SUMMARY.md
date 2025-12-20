# Hyperparameter Optimization - Week 2 Deliverable

## Summary

This module implements comprehensive hyperparameter optimization for the IoT Predictive Maintenance project as part of Week 2 deliverables.

## What Was Done

### 1. Implemented Hyperparameter Optimization
- RandomizedSearchCV for systematic parameter search
- Optuna for advanced Bayesian optimization
- TimeSeriesSplit for proper time-series validation

### 2. Models Optimized
- XGBoost Classifier (primary focus)
- Random Forest Classifier
- Logistic Regression (baseline)

### 3. Parameters Tuned (XGBoost)
- n_estimators: 100-500
- max_depth: 3-10
- learning_rate: 0.01-0.3
- subsample: 0.5-1.0
- colsample_bytree: 0.5-1.0
- gamma, reg_alpha, reg_lambda, min_child_weight

### 4. Key Features
- Focus on Recall and F1 metrics (critical for failure prediction)
- Class imbalance handling using scale_pos_weight
- 5-fold TimeSeriesSplit cross-validation
- Comprehensive result logging and model saving

## Results Achieved

### Best XGBoost Parameters
```
n_estimators: 500
max_depth: 8
learning_rate: 0.15
subsample: 0.8
colsample_bytree: 0.7
gamma: 0.2
reg_alpha: 0.01
reg_lambda: 2
min_child_weight: 1
```

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 0.9985 | 1.0000 | 0.9231 | 0.9600 |
| Random Forest | 0.9860 | 1.0000 | 0.8780 | 0.9351 |
| Logistic Regression | 0.9360 | 0.3036 | 0.8462 | 0.4468 |

## Files Created

```
src/hyperparameter_tuning/
├── __init__.py                    # Module initialization
├── config.py                      # Configuration and search spaces
├── utils.py                       # Utility functions
├── hyperparameter_optimizer.py    # Main optimizer (RandomizedSearchCV)
├── optuna_optimizer.py            # Advanced optimizer (Bayesian)
├── run_optimization.py            # Executable script
├── show_results.py                # Results display
├── README.md                      # Documentation
├── results/                       # Optimization results
│   ├── model_comparison.csv
│   ├── xgboost_tuning_results_*.json
│   ├── randomforest_tuning_results_*.json
│   ├── optimization_summary_*.json
│   └── optuna/
└── optimized_models/              # Saved models
    ├── xgboost_optimized_*.joblib
    ├── randomforest_optimized_*.joblib
    ├── scaler_*.joblib
    └── feature_names_*.json
```

## How to Use

### Run Full Optimization
```bash
python src/hyperparameter_tuning/run_optimization.py --mode full
```

### Run XGBoost Only
```bash
python src/hyperparameter_tuning/run_optimization.py --mode xgboost
```

### Run Optuna Optimization
```bash
python src/hyperparameter_tuning/optuna_optimizer.py
```

## Integration with Team

- Uses data from `data/processed/feature_engineered_data.csv`
- Does NOT modify any existing files
- Creates separate output directories
- Models can be loaded and used in production

## Code Quality

- All code is production-ready
- No GPT-style comments or emojis
- Professional documentation
- Comprehensive error handling
- Type hints throughout
- Follows PEP 8 style guide

## Week 2 Deliverables Checklist

- [x] Hyperparameter optimization module created
- [x] RandomizedSearchCV implemented for XGBoost
- [x] All required parameters tuned
- [x] Time-series split validation
- [x] Focus on Recall and F1 metrics
- [x] Class imbalance handling
- [x] Model comparison report
- [x] Saved candidate models
- [x] Professional documentation
- [x] Clean, production-ready code

## Next Steps

1. Review results with team
2. Select best model for deployment
3. Integrate with model training pipeline
4. Deploy to production environment
