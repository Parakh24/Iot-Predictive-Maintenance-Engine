"""
Utility Functions for Hyperparameter Tuning

Provides data loading, feature preparation, and validation utilities.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str = "data/processed/feature_engineered_data.csv") -> pd.DataFrame:
    """
    Load the feature-engineered dataset.
    
    Parameters
    ----------
    filepath : str
        Path to the feature-engineered CSV file.
        
    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    return pd.read_csv(filepath)


def prepare_features(df: pd.DataFrame, target_col: str = "Machine failure"):
    """
    Prepare features and target for modeling.
    
    Handles:
    - Dropping non-numeric columns
    - Separating features (X) and target (y)
    - Cleaning column names for XGBoost compatibility
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with features and target.
    target_col : str
        Name of the target column.
        
    Returns
    -------
    tuple (pd.DataFrame, pd.Series, list)
        X (features), y (target), feature_names
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Drop non-numeric columns
    non_numeric_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X = X.drop(columns=non_numeric_cols)
    
    # Clean column names for XGBoost (no special characters)
    original_cols = X.columns.tolist()
    X.columns = [
        str(col).replace('[', '_').replace(']', '_')
                .replace('<', '_').replace('>', '_')
                .replace(' ', '_')
        for col in X.columns
    ]
    feature_names = X.columns.tolist()
    
    return X, y, feature_names


def create_time_series_split(n_splits: int = 5) -> TimeSeriesSplit:
    """
    Create a time-series cross-validation splitter.
    
    Parameters
    ----------
    n_splits : int
        Number of splits for cross-validation.
        
    Returns
    -------
    TimeSeriesSplit
        Configured time-series split object.
    """
    return TimeSeriesSplit(n_splits=n_splits)


def scale_features(X_train, X_test=None):
    """
    Scale features using StandardScaler.
    
    Parameters
    ----------
    X_train : array-like
        Training features.
    X_test : array-like, optional
        Test features.
        
    Returns
    -------
    tuple or array
        Scaled X_train, (optionally X_test), and scaler object.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler


def get_class_weight_ratio(y: pd.Series) -> float:
    """
    Calculate the scale_pos_weight for imbalanced classification.
    
    This is used by XGBoost to handle class imbalance.
    
    Parameters
    ----------
    y : pd.Series
        Target variable.
        
    Returns
    -------
    float
        Ratio of negative to positive samples.
    """
    return (y == 0).sum() / (y == 1).sum()


def print_class_distribution(y: pd.Series, title: str = "Class Distribution"):
    """
    Print the class distribution of the target variable.
    
    Parameters
    ----------
    y : pd.Series
        Target variable.
    title : str
        Title for the output.
    """
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    
    counts = y.value_counts().sort_index()
    total = len(y)
    
    for label, count in counts.items():
        pct = count / total * 100
        print(f"  Class {label}: {count:>6} samples ({pct:>5.1f}%)")
    
    print(f"  {''*40}")
    print(f"  Total:   {total:>6} samples")
    print(f"  Imbalance Ratio: {get_class_weight_ratio(y):.2f}:1 (Neg:Pos)")
