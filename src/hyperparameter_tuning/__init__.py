"""
Hyperparameter Tuning Module for IoT Predictive Maintenance
Week 2: Modeling & Hyperparameter Tuning

This module provides comprehensive hyperparameter optimization using:
- RandomizedSearchCV for XGBoost
- Time-series split for proper validation
- Focus on Recall and F1 metrics
- Class imbalance handling

Author: Week 2 - Hyperparameter Optimization Team
"""

from .hyperparameter_optimizer import HyperparameterOptimizer
from .utils import load_data, prepare_features, create_time_series_split

__all__ = [
    'HyperparameterOptimizer',
    'load_data',
    'prepare_features',
    'create_time_series_split'
]
