"""
Hyperparameter Optimizer for XGBoost, Random Forest, and Logistic Regression

Uses RandomizedSearchCV with TimeSeriesSplit for proper time-series validation.
Focuses on Recall and F1 metrics as per Week 2 requirements.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score, f1_score, precision_score, accuracy_score,
    classification_report, confusion_matrix, make_scorer
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .utils import (
    load_data, prepare_features, create_time_series_split,
    scale_features, get_class_weight_ratio, print_class_distribution
)


class HyperparameterOptimizer:
    """
    Comprehensive Hyperparameter Optimization for Predictive Maintenance Models.
    
    Implements RandomizedSearchCV with TimeSeriesSplit for:
    - XGBoost Classifier
    - Random Forest Classifier  
    - Logistic Regression (Baseline)
    
    Focuses on Recall and F1 metrics for imbalanced failure prediction.
    """
    
    # Default hyperparameter search spaces
    XGBOOST_PARAM_GRID = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 4, 5, 6, 7, 8, 10],
        'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [1, 1.5, 2, 5]
    }
    
    RANDOM_FOREST_PARAM_GRID = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    LOGISTIC_PARAM_GRID = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000]
    }
    
    def __init__(
        self,
        data_path: str = "data/processed/feature_engineered_data.csv",
        output_dir: str = "src/hyperparameter_tuning/results",
        models_dir: str = "src/hyperparameter_tuning/optimized_models",
        n_cv_splits: int = 5,
        n_iter: int = 50,
        scoring: str = "f1",
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 2
    ):
        """
        Initialize the Hyperparameter Optimizer.
        
        Parameters
        ----------
        data_path : str
            Path to the feature-engineered data.
        output_dir : str
            Directory to save tuning results.
        models_dir : str
            Directory to save optimized models.
        n_cv_splits : int
            Number of cross-validation splits.
        n_iter : int
            Number of random parameter combinations to try.
        scoring : str
            Primary scoring metric ('f1', 'recall', 'precision', 'accuracy').
        random_state : int
            Random seed for reproducibility.
        n_jobs : int
            Number of parallel jobs (-1 for all cores).
        verbose : int
            Verbosity level for RandomizedSearchCV.
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.n_cv_splits = n_cv_splits
        self.n_iter = n_iter
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize data containers
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None
        
        # Results storage
        self.results = {}
        self.best_models = {}
        
    def load_and_prepare_data(self, test_size: float = 0.2):
        """
        Load data and prepare train/test splits using time-based splitting.
        
        Parameters
        ----------
        test_size : float
            Proportion of data to use for testing.
        """
        print("\n" + "="*60)
        print("📂 LOADING AND PREPARING DATA")
        print("="*60)
        
        # Load data
        df = load_data(self.data_path)
        print(f"✅ Loaded data from: {self.data_path}")
        print(f"   Shape: {df.shape}")
        
        # Prepare features
        self.X, self.y, self.feature_names = prepare_features(df)
        print(f"✅ Prepared {len(self.feature_names)} features")
        
        # Print class distribution
        print_class_distribution(self.y, "Target Class Distribution")
        
        # Time-based split (chronological ordering)
        split_idx = int(len(self.X) * (1 - test_size))
        self.X_train = self.X.iloc[:split_idx]
        self.X_test = self.X.iloc[split_idx:]
        self.y_train = self.y.iloc[:split_idx]
        self.y_test = self.y.iloc[split_idx:]
        
        print(f"\n📊 Time-Series Split:")
        print(f"   Train: {len(self.X_train)} samples")
        print(f"   Test:  {len(self.X_test)} samples")
        
        # Scale features
        X_train_scaled, X_test_scaled, self.scaler = scale_features(
            self.X_train, self.X_test
        )
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        
        print("✅ Features scaled using StandardScaler")
        
    def _get_cv_strategy(self):
        """Get the cross-validation strategy (TimeSeriesSplit)."""
        return create_time_series_split(n_splits=self.n_cv_splits)
    
    def _get_scorer(self):
        """Get the scoring function based on the scoring parameter."""
        scorers = {
            'f1': make_scorer(f1_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'precision': make_scorer(precision_score, zero_division=0),
            'accuracy': make_scorer(accuracy_score)
        }
        return scorers.get(self.scoring, scorers['f1'])
    
    def optimize_xgboost(
        self,
        param_grid: Optional[Dict] = None,
        n_iter: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize XGBoost classifier hyperparameters using RandomizedSearchCV.
        
        Parameters
        ----------
        param_grid : dict, optional
            Custom parameter grid. Uses default if None.
        n_iter : int, optional
            Number of iterations. Uses instance default if None.
            
        Returns
        -------
        dict
            Dictionary containing best parameters, scores, and model.
        """
        print("\n" + "="*60)
        print("🚀 OPTIMIZING XGBOOST CLASSIFIER")
        print("="*60)
        
        param_grid = param_grid or self.XGBOOST_PARAM_GRID
        n_iter = n_iter or self.n_iter
        
        # Calculate class weight for imbalance handling
        scale_pos_weight = get_class_weight_ratio(self.y_train)
        print(f"📊 Scale Pos Weight (for imbalance): {scale_pos_weight:.2f}")
        
        # Base model with imbalance handling
        base_model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=self.random_state,
            use_label_encoder=False
        )
        
        # RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=self._get_scorer(),
            cv=self._get_cv_strategy(),
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            return_train_score=True
        )
        
        print(f"\n🔍 Starting RandomizedSearchCV...")
        print(f"   Iterations: {n_iter}")
        print(f"   CV Splits: {self.n_cv_splits}")
        print(f"   Scoring: {self.scoring}")
        
        search.fit(self.X_train_scaled, self.y_train)
        
        # Store results
        result = self._evaluate_and_store(search, 'XGBoost')
        return result
    
    def optimize_random_forest(
        self,
        param_grid: Optional[Dict] = None,
        n_iter: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize Random Forest classifier hyperparameters.
        
        Parameters
        ----------
        param_grid : dict, optional
            Custom parameter grid.
        n_iter : int, optional
            Number of iterations.
            
        Returns
        -------
        dict
            Dictionary containing best parameters, scores, and model.
        """
        print("\n" + "="*60)
        print("🌲 OPTIMIZING RANDOM FOREST CLASSIFIER")
        print("="*60)
        
        param_grid = param_grid or self.RANDOM_FOREST_PARAM_GRID
        n_iter = n_iter or self.n_iter
        
        # Base model with class weight balancing
        base_model = RandomForestClassifier(
            class_weight='balanced',
            random_state=self.random_state
        )
        
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=self._get_scorer(),
            cv=self._get_cv_strategy(),
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            return_train_score=True
        )
        
        print(f"\n🔍 Starting RandomizedSearchCV...")
        print(f"   Iterations: {n_iter}")
        print(f"   CV Splits: {self.n_cv_splits}")
        print(f"   Scoring: {self.scoring}")
        
        search.fit(self.X_train_scaled, self.y_train)
        
        result = self._evaluate_and_store(search, 'RandomForest')
        return result
    
    def optimize_logistic_regression(
        self,
        param_grid: Optional[Dict] = None,
        n_iter: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize Logistic Regression (baseline) hyperparameters.
        
        Parameters
        ----------
        param_grid : dict, optional
            Custom parameter grid.
        n_iter : int, optional
            Number of iterations.
            
        Returns
        -------
        dict
            Dictionary containing best parameters, scores, and model.
        """
        print("\n" + "="*60)
        print("📈 OPTIMIZING LOGISTIC REGRESSION (BASELINE)")
        print("="*60)
        
        param_grid = param_grid or self.LOGISTIC_PARAM_GRID
        n_iter = n_iter or min(self.n_iter, 20)  # Fewer iterations for baseline
        
        base_model = LogisticRegression(
            class_weight='balanced',
            random_state=self.random_state
        )
        
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=self._get_scorer(),
            cv=self._get_cv_strategy(),
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            return_train_score=True
        )
        
        print(f"\n🔍 Starting RandomizedSearchCV...")
        print(f"   Iterations: {n_iter}")
        print(f"   CV Splits: {self.n_cv_splits}")
        print(f"   Scoring: {self.scoring}")
        
        search.fit(self.X_train_scaled, self.y_train)
        
        result = self._evaluate_and_store(search, 'LogisticRegression')
        return result
    
    def _evaluate_and_store(
        self,
        search: RandomizedSearchCV,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate the best model and store results.
        
        Parameters
        ----------
        search : RandomizedSearchCV
            Completed search object.
        model_name : str
            Name of the model.
            
        Returns
        -------
        dict
            Evaluation results.
        """
        best_model = search.best_estimator_
        y_pred = best_model.predict(self.X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0)
        }
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Store result
        result = {
            'model_name': model_name,
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'test_metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(
                self.y_test, y_pred, output_dict=True, zero_division=0
            ),
            'cv_results_summary': {
                'mean_train_score': search.cv_results_['mean_train_score'].tolist(),
                'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': search.cv_results_['std_test_score'].tolist()
            }
        }
        
        self.results[model_name] = result
        self.best_models[model_name] = best_model
        
        # Print results
        self._print_results(model_name, result)
        
        return result
    
    def _print_results(self, model_name: str, result: Dict[str, Any]):
        """Print formatted results for a model."""
        print(f"\n{'─'*50}")
        print(f"✅ {model_name} OPTIMIZATION COMPLETE")
        print(f"{'─'*50}")
        
        print("\n📋 Best Parameters:")
        for param, value in result['best_params'].items():
            print(f"   {param}: {value}")
        
        print(f"\n📊 Cross-Validation Score ({self.scoring}):")
        print(f"   CV Score: {result['best_cv_score']:.4f}")
        
        print("\n🎯 Test Set Performance:")
        for metric, value in result['test_metrics'].items():
            print(f"   {metric.capitalize()}: {value:.4f}")
        
        print("\n📉 Confusion Matrix:")
        cm = result['confusion_matrix']
        print(f"   TN: {cm[0][0]:>5}  |  FP: {cm[0][1]:>5}")
        print(f"   FN: {cm[1][0]:>5}  |  TP: {cm[1][1]:>5}")
    
    def run_full_optimization(self) -> Dict[str, Any]:
        """
        Run full hyperparameter optimization for all models.
        
        Returns
        -------
        dict
            Results for all models and comparison report.
        """
        print("\n" + "="*70)
        print("🎯 STARTING FULL HYPERPARAMETER OPTIMIZATION PIPELINE")
        print("="*70)
        print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        self.load_and_prepare_data()
        
        # Optimize all models
        self.optimize_logistic_regression()
        self.optimize_random_forest()
        self.optimize_xgboost()
        
        # Generate comparison report
        comparison = self.generate_comparison_report()
        
        # Save all results
        self.save_results()
        self.save_models()
        
        return {
            'results': self.results,
            'comparison': comparison
        }
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """
        Generate a comparison report of all optimized models.
        
        Returns
        -------
        pd.DataFrame
            Comparison dataframe sorted by F1 score.
        """
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON REPORT")
        print("="*60)
        
        comparison_data = []
        for model_name, result in self.results.items():
            row = {
                'Model': model_name,
                'CV_Score': result['best_cv_score'],
                'Accuracy': result['test_metrics']['accuracy'],
                'Precision': result['test_metrics']['precision'],
                'Recall': result['test_metrics']['recall'],
                'F1_Score': result['test_metrics']['f1']
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1_Score', ascending=False)
        
        print("\n" + comparison_df.to_string(index=False))
        
        # Find best model
        best = comparison_df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best['Model']}")
        print(f"   F1 Score: {best['F1_Score']:.4f}")
        print(f"   Recall:   {best['Recall']:.4f}")
        
        # Save comparison
        comparison_path = os.path.join(self.output_dir, 'model_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\n📁 Comparison saved to: {comparison_path}")
        
        return comparison_df
    
    def save_results(self):
        """Save all optimization results to JSON files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results for each model
        for model_name, result in self.results.items():
            # Create serializable version
            result_to_save = {
                'model_name': result['model_name'],
                'best_params': {k: str(v) if not isinstance(v, (int, float, bool, type(None))) else v 
                               for k, v in result['best_params'].items()},
                'best_cv_score': float(result['best_cv_score']),
                'test_metrics': {k: float(v) for k, v in result['test_metrics'].items()},
                'confusion_matrix': result['confusion_matrix'],
                'timestamp': timestamp
            }
            
            filename = f"{model_name.lower()}_tuning_results_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(result_to_save, f, indent=2)
            
            print(f"📁 Saved: {filepath}")
        
        # Save summary of all results
        summary = {
            'timestamp': timestamp,
            'scoring_metric': self.scoring,
            'n_cv_splits': self.n_cv_splits,
            'n_iterations': self.n_iter,
            'models': list(self.results.keys()),
            'best_scores': {
                name: {
                    'cv_score': float(res['best_cv_score']),
                    'f1': float(res['test_metrics']['f1']),
                    'recall': float(res['test_metrics']['recall'])
                }
                for name, res in self.results.items()
            }
        }
        
        summary_path = os.path.join(self.output_dir, f'optimization_summary_{timestamp}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📁 Saved: {summary_path}")
    
    def save_models(self):
        """Save all optimized models and the scaler."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save models
        for model_name, model in self.best_models.items():
            filename = f"{model_name.lower()}_optimized_{timestamp}.joblib"
            filepath = os.path.join(self.models_dir, filename)
            joblib.dump(model, filepath)
            print(f"📁 Saved model: {filepath}")
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, f'scaler_{timestamp}.joblib')
        joblib.dump(self.scaler, scaler_path)
        print(f"📁 Saved scaler: {scaler_path}")
        
        # Save feature names
        features_path = os.path.join(self.models_dir, f'feature_names_{timestamp}.json')
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"📁 Saved feature names: {features_path}")
    
    def get_best_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get the best hyperparameters for a specific model.
        
        Parameters
        ----------
        model_name : str
            Name of the model ('XGBoost', 'RandomForest', 'LogisticRegression').
            
        Returns
        -------
        dict
            Best hyperparameters.
        """
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.results.keys())}")
        return self.results[model_name]['best_params']
    
    def get_best_model(self, model_name: str = None):
        """
        Get the best model(s).
        
        Parameters
        ----------
        model_name : str, optional
            Name of specific model. Returns all if None.
            
        Returns
        -------
        model or dict
            Best model or dictionary of all models.
        """
        if model_name:
            if model_name not in self.best_models:
                raise ValueError(f"Model '{model_name}' not found. Available: {list(self.best_models.keys())}")
            return self.best_models[model_name]
        return self.best_models
