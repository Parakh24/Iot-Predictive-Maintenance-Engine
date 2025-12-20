"""
Final Results Summary Script
Displays both RandomizedSearchCV and Optuna optimization results
"""
import json
import os

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def main():
    print("""
    
          HYPERPARAMETER OPTIMIZATION - FINAL RESULTS SUMMARY       
                      Week 2: IoT Predictive Maintenance              
    
    """)

    # ========================================
    # RandomizedSearchCV Results
    # ========================================
    print_section(" RANDOMIZEDSEARCHCV RESULTS")
    
    results_dir = "src/hyperparameter_tuning/results"
    
    # Find and load XGBoost results
    for f in os.listdir(results_dir):
        if f.startswith('xgboost_tuning') and f.endswith('.json'):
            with open(os.path.join(results_dir, f)) as file:
                xgb = json.load(file)
            
            print("\n XGBoost (RandomizedSearchCV):")
            print("   Best Parameters:")
            for param, value in xgb['best_params'].items():
                print(f"       {param}: {value}")
            
            print("\n    Performance Metrics:")
            print(f"       CV Score:  {xgb['best_cv_score']:.4f}")
            print(f"       Accuracy:  {xgb['test_metrics']['accuracy']:.4f}")
            print(f"       Precision: {xgb['test_metrics']['precision']:.4f}")
            print(f"       Recall:    {xgb['test_metrics']['recall']:.4f}")
            print(f"       F1 Score:  {xgb['test_metrics']['f1']:.4f}")
            break
    
    # Find and load Random Forest results
    for f in os.listdir(results_dir):
        if f.startswith('randomforest_tuning') and f.endswith('.json'):
            with open(os.path.join(results_dir, f)) as file:
                rf = json.load(file)
            
            print("\n Random Forest (RandomizedSearchCV):")
            print("   Best Parameters:")
            for param, value in rf['best_params'].items():
                print(f"       {param}: {value}")
            
            print("\n    Performance Metrics:")
            print(f"       CV Score:  {rf['best_cv_score']:.4f}")
            print(f"       Accuracy:  {rf['test_metrics']['accuracy']:.4f}")
            print(f"       Precision: {rf['test_metrics']['precision']:.4f}")
            print(f"       Recall:    {rf['test_metrics']['recall']:.4f}")
            print(f"       F1 Score:  {rf['test_metrics']['f1']:.4f}")
            break

    # ========================================
    # Optuna Results
    # ========================================
    optuna_dir = "src/hyperparameter_tuning/results/optuna"
    
    if os.path.exists(optuna_dir):
        print_section(" OPTUNA BAYESIAN OPTIMIZATION RESULTS")
        
        for f in os.listdir(optuna_dir):
            if f.startswith('xgboost_optuna') and f.endswith('.json'):
                with open(os.path.join(optuna_dir, f)) as file:
                    xgb_opt = json.load(file)
                
                print("\n XGBoost (Optuna - Bayesian):")
                print("   Best Parameters:")
                for param, value in xgb_opt['best_params'].items():
                    if isinstance(value, float):
                        print(f"       {param}: {value:.6f}")
                    else:
                        print(f"       {param}: {value}")
                
                print("\n    Performance Metrics:")
                print(f"       CV Score:    {xgb_opt['best_cv_score']:.4f}")
                print(f"       Test F1:     {xgb_opt['test_f1']:.4f}")
                print(f"       Test Recall: {xgb_opt['test_recall']:.4f}")
                print(f"       Trials:      {xgb_opt['n_trials']}")
                break
        
        for f in os.listdir(optuna_dir):
            if f.startswith('randomforest_optuna') and f.endswith('.json'):
                with open(os.path.join(optuna_dir, f)) as file:
                    rf_opt = json.load(file)
                
                print("\n Random Forest (Optuna - Bayesian):")
                print("   Best Parameters:")
                for param, value in rf_opt['best_params'].items():
                    print(f"       {param}: {value}")
                
                print("\n    Performance Metrics:")
                print(f"       CV Score:    {rf_opt['best_cv_score']:.4f}")
                print(f"       Test F1:     {rf_opt['test_f1']:.4f}")
                print(f"       Test Recall: {rf_opt['test_recall']:.4f}")
                break

    # ========================================
    # Saved Models
    # ========================================
    print_section(" SAVED OPTIMIZED MODELS")
    
    models_dir = "src/hyperparameter_tuning/optimized_models"
    if os.path.exists(models_dir):
        for f in sorted(os.listdir(models_dir)):
            size_kb = os.path.getsize(os.path.join(models_dir, f)) / 1024
            print(f"    {f} ({size_kb:.1f} KB)")
    
    # ========================================
    # Week 2 Deliverables Check
    # ========================================
    print_section(" WEEK 2 DELIVERABLES CHECKLIST")
    
    print("""
    [] RandomizedSearchCV for XGBoost implemented
    [] Parameters tuned:
         n_estimators
         max_depth
         learning_rate
         subsample
         colsample_bytree
    [] Time-series split validation (TimeSeriesSplit)
    [] Focus on Recall and F1 metrics
    [] Class imbalance handling (scale_pos_weight, class_weight)
    [] Model comparison report generated
    [] Saved candidate models (.joblib files)
    [] Feature scaler saved
    [] Feature names saved
    """)

    print("\n" + "="*70)
    print("   HYPERPARAMETER OPTIMIZATION COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
