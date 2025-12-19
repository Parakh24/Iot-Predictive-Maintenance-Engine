"""
Configuration for Hyperparameter Tuning Module

This file contains all configurable parameters for the hyperparameter optimization.
Modify these values to customize the optimization behavior.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# DATA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DATA_CONFIG = {
    "data_path": "data/processed/feature_engineered_data.csv",
    "target_column": "Machine failure",
    "test_size": 0.2,
}

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

OPTIMIZATION_CONFIG = {
    # Cross-validation settings
    "n_cv_splits": 5,               # Number of TimeSeriesSplit folds
    
    # RandomizedSearchCV settings
    "n_iter": 50,                   # Number of random parameter combinations
    
    # Optuna settings
    "n_trials": 100,                # Number of Optuna trials
    "optuna_timeout": None,         # Timeout in seconds (None = no timeout)
    
    # Scoring metric (focus on F1 and Recall as per Week 2 requirements)
    "primary_metric": "f1",         # Options: 'f1', 'recall', 'precision', 'accuracy'
    
    # Reproducibility
    "random_state": 42,
    
    # Parallel processing
    "n_jobs": -1,                   # -1 = use all CPU cores
    
    # Output
    "verbose": 1,                   # Verbosity level (0, 1, or 2)
}

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT PATHS
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_CONFIG = {
    "results_dir": "src/hyperparameter_tuning/results",
    "models_dir": "src/hyperparameter_tuning/optimized_models",
    "optuna_results_dir": "src/hyperparameter_tuning/results/optuna",
}

# ═══════════════════════════════════════════════════════════════════════════════
# XGBOOST HYPERPARAMETER SEARCH SPACE
# (Week 2 Focus: n_estimators, max_depth, learning_rate, subsample, colsample_bytree)
# ═══════════════════════════════════════════════════════════════════════════════

XGBOOST_PARAM_GRID = {
    # Number of boosting rounds
    "n_estimators": [100, 200, 300, 400, 500],
    
    # Maximum tree depth for base learners
    "max_depth": [3, 4, 5, 6, 7, 8, 10],
    
    # Boosting learning rate
    "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
    
    # Subsample ratio of the training instance
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    
    # Subsample ratio of columns when constructing each tree
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    
    # Minimum sum of instance weight needed in a child
    "min_child_weight": [1, 3, 5, 7],
    
    # Minimum loss reduction required to make a split
    "gamma": [0, 0.1, 0.2, 0.3],
    
    # L1 regularization term on weights
    "reg_alpha": [0, 0.01, 0.1, 1],
    
    # L2 regularization term on weights
    "reg_lambda": [1, 1.5, 2, 5],
}

# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST HYPERPARAMETER SEARCH SPACE
# ═══════════════════════════════════════════════════════════════════════════════

RANDOM_FOREST_PARAM_GRID = {
    # Number of trees in the forest
    "n_estimators": [100, 200, 300, 400, 500],
    
    # Maximum depth of the tree
    "max_depth": [5, 10, 15, 20, None],
    
    # Minimum samples required to split a node
    "min_samples_split": [2, 5, 10, 15],
    
    # Minimum samples required at each leaf node
    "min_samples_leaf": [1, 2, 4, 8],
    
    # Number of features to consider for best split
    "max_features": ["sqrt", "log2", None],
    
    # Whether bootstrap samples are used
    "bootstrap": [True, False],
}

# ═══════════════════════════════════════════════════════════════════════════════
# LOGISTIC REGRESSION HYPERPARAMETER SEARCH SPACE (BASELINE)
# ═══════════════════════════════════════════════════════════════════════════════

LOGISTIC_PARAM_GRID = {
    # Inverse of regularization strength
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    
    # Penalty norm
    "penalty": ["l1", "l2"],
    
    # Algorithm to use in the optimization
    "solver": ["liblinear", "saga"],
    
    # Maximum iterations for solver convergence
    "max_iter": [1000, 2000],
}

# ═══════════════════════════════════════════════════════════════════════════════
# OPTUNA-SPECIFIC SEARCH RANGES (Continuous parameters for Bayesian optimization)
# ═══════════════════════════════════════════════════════════════════════════════

OPTUNA_XGBOOST_RANGES = {
    "n_estimators": {"low": 100, "high": 500, "type": "int"},
    "max_depth": {"low": 3, "high": 10, "type": "int"},
    "learning_rate": {"low": 0.01, "high": 0.3, "type": "float", "log": True},
    "subsample": {"low": 0.5, "high": 1.0, "type": "float"},
    "colsample_bytree": {"low": 0.5, "high": 1.0, "type": "float"},
    "min_child_weight": {"low": 1, "high": 10, "type": "int"},
    "gamma": {"low": 0, "high": 0.5, "type": "float"},
    "reg_alpha": {"low": 1e-8, "high": 10.0, "type": "float", "log": True},
    "reg_lambda": {"low": 1e-8, "high": 10.0, "type": "float", "log": True},
}

OPTUNA_RANDOM_FOREST_RANGES = {
    "n_estimators": {"low": 100, "high": 500, "type": "int"},
    "max_depth": {"low": 5, "high": 30, "type": "int"},
    "min_samples_split": {"low": 2, "high": 20, "type": "int"},
    "min_samples_leaf": {"low": 1, "high": 10, "type": "int"},
    "max_features": {"options": ["sqrt", "log2", None], "type": "categorical"},
}
