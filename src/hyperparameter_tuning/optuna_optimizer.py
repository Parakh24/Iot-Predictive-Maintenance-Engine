"""
Advanced Hyperparameter Optimization using Optuna

Optuna provides more sophisticated optimization strategies compared to RandomizedSearchCV:
- Bayesian optimization (Tree-structured Parzen Estimator)
- Pruning of unpromising trials
- Better exploration of the parameter space

This complements the RandomizedSearchCV approach for Week 2.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not installed. Install with: pip install optuna")

from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import f1_score, recall_score, make_scorer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.hyperparameter_tuning.utils import (
    load_data, prepare_features, scale_features, 
    get_class_weight_ratio, print_class_distribution
)


class OptunaOptimizer:
    """
    Advanced hyperparameter optimization using Optuna's Bayesian optimization.
    
    Features:
    - TPE (Tree-structured Parzen Estimator) sampler
    - Pruning of unpromising trials
    - Early stopping integration
    - Visualization of optimization history
    """
    
    def __init__(
        self,
        data_path: str = "data/processed/feature_engineered_data.csv",
        output_dir: str = "src/hyperparameter_tuning/results/optuna",
        models_dir: str = "src/hyperparameter_tuning/optimized_models",
        n_trials: int = 100,
        n_cv_splits: int = 5,
        scoring: str = "f1",
        random_state: int = 42,
        timeout: int = None
    ):
        """
        Initialize the Optuna optimizer.
        
        Parameters
        ----------
        data_path : str
            Path to the feature-engineered data.
        output_dir : str
            Directory to save optimization results.
        models_dir : str
            Directory to save optimized models.
        n_trials : int
            Number of optimization trials.
        n_cv_splits : int
            Number of CV splits for TimeSeriesSplit.
        scoring : str
            Scoring metric ('f1' or 'recall').
        random_state : int
            Random seed for reproducibility.
        timeout : int, optional
            Timeout in seconds for the optimization.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")
        
        self.data_path = data_path
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.scoring = scoring
        self.random_state = random_state
        self.timeout = timeout
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Data containers
        self.X_train_scaled = None
        self.y_train = None
        self.X_test_scaled = None
        self.y_test = None
        self.scaler = None
        self.feature_names = None
        self.scale_pos_weight = None
        
        # Results
        self.studies = {}
        self.best_models = {}
        self.results = {}
    
    def load_and_prepare_data(self, test_size: float = 0.2):
        """Load and prepare data with time-based split."""
        print("\n" + "="*60)
        print("LOADING AND PREPARING DATA FOR OPTUNA OPTIMIZATION")
        print("="*60)
        
        df = load_data(self.data_path)
        X, y, self.feature_names = prepare_features(df)
        
        print_class_distribution(y, "Target Distribution")
        
        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        self.y_train, self.y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale
        self.X_train_scaled, self.X_test_scaled, self.scaler = scale_features(X_train, X_test)
        
        # Calculate class imbalance weight
        self.scale_pos_weight = get_class_weight_ratio(self.y_train)
        
        print(f"Data loaded. Train: {len(self.y_train)}, Test: {len(self.y_test)}")
        print(f"   Scale pos weight: {self.scale_pos_weight:.2f}")
    
    def _get_cv(self):
        """Get time-series cross-validation splitter."""
        return TimeSeriesSplit(n_splits=self.n_cv_splits)
    
    def _get_scorer(self):
        """Get the scorer function."""
        if self.scoring == "recall":
            return make_scorer(recall_score, zero_division=0)
        return make_scorer(f1_score, zero_division=0)
    
    def _xgboost_objective(self, trial: 'optuna.Trial') -> float:
        """Objective function for XGBoost optimization."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': self.scale_pos_weight,
            'eval_metric': 'logloss',
            'random_state': self.random_state,
            'use_label_encoder': False
        }
        
        model = XGBClassifier(**params)
        
        # Cross-validation score
        scores = cross_val_score(
            model, self.X_train_scaled, self.y_train,
            cv=self._get_cv(),
            scoring=self._get_scorer(),
            n_jobs=-1
        )
        
        return scores.mean()
    
    def _random_forest_objective(self, trial: 'optuna.Trial') -> float:
        """Objective function for Random Forest optimization."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': 'balanced',
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        model = RandomForestClassifier(**params)
        
        scores = cross_val_score(
            model, self.X_train_scaled, self.y_train,
            cv=self._get_cv(),
            scoring=self._get_scorer(),
            n_jobs=-1
        )
        
        return scores.mean()
    
    def optimize_xgboost(self) -> Dict[str, Any]:
        """
        Optimize XGBoost using Optuna's Bayesian optimization.
        
        Returns
        -------
        dict
            Best parameters and scores.
        """
        print("\n" + "="*60)
        print("OPTUNA XGBOOST OPTIMIZATION (Bayesian)")
        print("="*60)
        
        # Create study
        sampler = TPESampler(seed=self.random_state)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name='xgboost_optimization'
        )
        
        # Optimize
        study.optimize(
            self._xgboost_objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        self.studies['XGBoost'] = study
        
        # Train final model with best params
        best_params = study.best_params
        best_params['scale_pos_weight'] = self.scale_pos_weight
        best_params['eval_metric'] = 'logloss'
        best_params['random_state'] = self.random_state
        best_params['use_label_encoder'] = False
        
        best_model = XGBClassifier(**best_params)
        best_model.fit(self.X_train_scaled, self.y_train)
        
        self.best_models['XGBoost'] = best_model
        
        # Evaluate
        y_pred = best_model.predict(self.X_test_scaled)
        result = {
            'model_name': 'XGBoost',
            'best_params': study.best_params,
            'best_cv_score': study.best_value,
            'test_f1': f1_score(self.y_test, y_pred, zero_division=0),
            'test_recall': recall_score(self.y_test, y_pred, zero_division=0),
            'n_trials': len(study.trials)
        }
        
        self.results['XGBoost'] = result
        
        print(f"\nBest {self.scoring}: {study.best_value:.4f}")
        print(f"   Test F1: {result['test_f1']:.4f}")
        print(f"   Test Recall: {result['test_recall']:.4f}")
        print(f"\nBest Parameters:")
        for k, v in study.best_params.items():
            print(f"   {k}: {v}")
        
        return result
    
    def optimize_random_forest(self) -> Dict[str, Any]:
        """
        Optimize Random Forest using Optuna.
        
        Returns
        -------
        dict
            Best parameters and scores.
        """
        print("\n" + "="*60)
        print("OPTUNA RANDOM FOREST OPTIMIZATION (Bayesian)")
        print("="*60)
        
        sampler = TPESampler(seed=self.random_state)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='rf_optimization'
        )
        
        study.optimize(
            self._random_forest_objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        self.studies['RandomForest'] = study
        
        # Train final model
        best_params = study.best_params
        best_params['class_weight'] = 'balanced'
        best_params['random_state'] = self.random_state
        best_params['n_jobs'] = -1
        
        best_model = RandomForestClassifier(**best_params)
        best_model.fit(self.X_train_scaled, self.y_train)
        
        self.best_models['RandomForest'] = best_model
        
        # Evaluate
        y_pred = best_model.predict(self.X_test_scaled)
        result = {
            'model_name': 'RandomForest',
            'best_params': study.best_params,
            'best_cv_score': study.best_value,
            'test_f1': f1_score(self.y_test, y_pred, zero_division=0),
            'test_recall': recall_score(self.y_test, y_pred, zero_division=0),
            'n_trials': len(study.trials)
        }
        
        self.results['RandomForest'] = result
        
        print(f"\nBest {self.scoring}: {study.best_value:.4f}")
        print(f"   Test F1: {result['test_f1']:.4f}")
        print(f"   Test Recall: {result['test_recall']:.4f}")
        
        return result
    
    def run_full_optimization(self) -> Dict[str, Any]:
        """Run full Optuna optimization for all models."""
        print("""
        ╔══════════════════════════════════════════════════════════════════╗
        ║             OPTUNA HYPERPARAMETER OPTIMIZATION                   ║
        ║              (Advanced Bayesian Optimization)                    ║
        ╚══════════════════════════════════════════════════════════════════╝
        """)
        
        self.load_and_prepare_data()
        
        self.optimize_random_forest()
        self.optimize_xgboost()
        
        self.save_results()
        self.save_models()
        
        # Comparison
        self.generate_comparison()
        
        return self.results
    
    def generate_comparison(self):
        """Generate and print comparison of models."""
        print("\n" + "="*60)
        print("OPTUNA OPTIMIZATION COMPARISON")
        print("="*60)
        
        data = []
        for name, result in self.results.items():
            data.append({
                'Model': name,
                'CV_Score': result['best_cv_score'],
                'Test_F1': result['test_f1'],
                'Test_Recall': result['test_recall'],
                'N_Trials': result['n_trials']
            })
        
        df = pd.DataFrame(data).sort_values('Test_F1', ascending=False)
        print(df.to_string(index=False))
        
        # Save comparison
        df.to_csv(os.path.join(self.output_dir, 'optuna_comparison.csv'), index=False)
    
    def save_results(self):
        """Save optimization results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, result in self.results.items():
            result_to_save = {
                'model_name': result['model_name'],
                'best_params': {k: str(v) if isinstance(v, type(None)) else v 
                               for k, v in result['best_params'].items()},
                'best_cv_score': float(result['best_cv_score']),
                'test_f1': float(result['test_f1']),
                'test_recall': float(result['test_recall']),
                'n_trials': result['n_trials'],
                'timestamp': timestamp
            }
            
            filepath = os.path.join(self.output_dir, f'{name.lower()}_optuna_{timestamp}.json')
            with open(filepath, 'w') as f:
                json.dump(result_to_save, f, indent=2)
            print(f"Saved: {filepath}")
    
    def save_models(self):
        """Save optimized models."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, model in self.best_models.items():
            filepath = os.path.join(self.models_dir, f'{name.lower()}_optuna_{timestamp}.joblib')
            joblib.dump(model, filepath)
            print(f"Saved model: {filepath}")
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, f'scaler_optuna_{timestamp}.joblib')
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved scaler: {scaler_path}")
    
    def visualize_optimization(self, model_name: str = 'XGBoost'):
        """
        Create visualization of the optimization history.
        
        Parameters
        ----------
        model_name : str
            Name of the model to visualize.
        """
        if model_name not in self.studies:
            print(f"No study found for {model_name}")
            return
        
        study = self.studies[model_name]
        
        try:
            # Import visualization modules
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_contour
            )
            
            # Optimization history
            fig1 = plot_optimization_history(study)
            fig1.write_html(os.path.join(self.output_dir, f'{model_name.lower()}_history.html'))
            
            # Parameter importances
            fig2 = plot_param_importances(study)
            fig2.write_html(os.path.join(self.output_dir, f'{model_name.lower()}_importance.html'))
            
            print(f"Visualizations saved to {self.output_dir}")
            
        except Exception as e:
            print(f"Could not create visualizations: {e}")


def main():
    """Run Optuna optimization."""
    optimizer = OptunaOptimizer(
        n_trials=50,
        n_cv_splits=5,
        scoring='f1',
        random_state=42
    )
    
    results = optimizer.run_full_optimization()
    
    # Try to create visualizations
    optimizer.visualize_optimization('XGBoost')
    
    return results


if __name__ == "__main__":
    main()
