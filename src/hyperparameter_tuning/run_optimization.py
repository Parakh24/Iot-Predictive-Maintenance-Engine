"""
Run Hyperparameter Optimization Pipeline

This script runs the complete hyperparameter optimization for:
- Logistic Regression (Baseline)
- Random Forest Classifier
- XGBoost Classifier

Usage:
    python -m src.hyperparameter_tuning.run_optimization
    
Or from project root:
    python src/hyperparameter_tuning/run_optimization.py
"""

import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.hyperparameter_tuning.hyperparameter_optimizer import HyperparameterOptimizer


def main():
    """Run the full hyperparameter optimization pipeline."""
    
    print("""
    
            IoT PREDICTIVE MAINTENANCE - HYPERPARAMETER TUNING      
                            Week 2 Deliverable                         
    
    """)
    
    # Initialize optimizer with configuration
    optimizer = HyperparameterOptimizer(
        data_path="data/processed/feature_engineered_data.csv",
        output_dir="src/hyperparameter_tuning/results",
        models_dir="src/hyperparameter_tuning/optimized_models",
        n_cv_splits=5,           # 5-fold Time Series Split
        n_iter=50,               # Number of random combinations to try
        scoring="f1",            # Optimize for F1 (balances precision & recall)
        random_state=42,         # For reproducibility
        n_jobs=-1,               # Use all CPU cores
        verbose=1                # Show progress
    )
    
    # Run full optimization
    results = optimizer.run_full_optimization()
    
    print("""
    
                         OPTIMIZATION COMPLETE                       
                                                                      
       Results saved to:  src/hyperparameter_tuning/results/          
       Models saved to:   src/hyperparameter_tuning/optimized_models/ 
    
    """)
    
    return results


def run_quick_optimization():
    """Run a quick optimization with fewer iterations (for testing)."""
    
    print("""
    
            QUICK HYPERPARAMETER OPTIMIZATION (Testing Mode)        
    
    """)
    
    optimizer = HyperparameterOptimizer(
        data_path="data/processed/feature_engineered_data.csv",
        output_dir="src/hyperparameter_tuning/results",
        models_dir="src/hyperparameter_tuning/optimized_models",
        n_cv_splits=3,           # 3-fold for speed
        n_iter=10,               # Fewer iterations for testing
        scoring="f1",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    results = optimizer.run_full_optimization()
    return results


def run_xgboost_only():
    """Optimize only XGBoost (main focus as per Week 2 requirements)."""
    
    print("""
    
                   XGBOOST HYPERPARAMETER OPTIMIZATION              
                  (Primary Model - Week 2 Focus)                      
    
    """)
    
    optimizer = HyperparameterOptimizer(
        data_path="data/processed/feature_engineered_data.csv",
        output_dir="src/hyperparameter_tuning/results",
        models_dir="src/hyperparameter_tuning/optimized_models",
        n_cv_splits=5,
        n_iter=100,              # More iterations for XGBoost focus
        scoring="f1",
        random_state=42,
        n_jobs=-1,
        verbose=2
    )
    
    # Load data
    optimizer.load_and_prepare_data()
    
    # Optimize only XGBoost
    result = optimizer.optimize_xgboost()
    
    # Save results
    optimizer.save_results()
    optimizer.save_models()
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization for Predictive Maintenance')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'quick', 'xgboost'],
                        help='Optimization mode: full (all models), quick (testing), xgboost (only XGBoost)')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        main()
    elif args.mode == 'quick':
        run_quick_optimization()
    elif args.mode == 'xgboost':
        run_xgboost_only()
