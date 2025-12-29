import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib


class SHAPExplainer:
    
    def __init__(self, model, X_data, feature_names=None):
        self.model = model
        self.X_data = X_data
        self.feature_names = feature_names if feature_names is not None else list(X_data.columns)
        self.explainer = None
        self.shap_values = None
        
    def compute_shap_values(self):
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(self.X_data)
        return self.shap_values
    
    def get_shap_values(self):
        if self.shap_values is None:
            self.compute_shap_values()
        return self.shap_values
    
    def summary_plot(self, output_path=None, max_display=20):
        if self.shap_values is None:
            self.compute_shap_values()
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_data,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title("SHAP Summary Plot - Feature Impact on Predictions", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Summary plot saved to: {output_path}")
        plt.close()
        
    def bar_plot(self, output_path=None, max_display=20):
        if self.shap_values is None:
            self.compute_shap_values()
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_data,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        plt.title("SHAP Bar Plot - Global Feature Importance", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Bar plot saved to: {output_path}")
        plt.close()
        
    def force_plot(self, instance_index=0, output_path=None):
        if self.shap_values is None:
            self.compute_shap_values()
        
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        
        instance_shap = shap_vals[instance_index]
        instance_features = self.X_data.iloc[instance_index]
        
        plt.figure(figsize=(20, 4))
        shap.force_plot(
            expected_value,
            instance_shap,
            instance_features,
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f"Force Plot - Local Explanation for Machine Instance {instance_index}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Force plot saved to: {output_path}")
        plt.close()
        
    def decision_plot(self, instance_indices=None, output_path=None, max_display=15):
        if self.shap_values is None:
            self.compute_shap_values()
        
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        
        if instance_indices is None:
            instance_indices = list(range(min(10, len(self.X_data))))
        
        plt.figure(figsize=(12, 10))
        shap.decision_plot(
            expected_value,
            shap_vals[instance_indices],
            self.X_data.iloc[instance_indices],
            feature_names=self.feature_names,
            show=False
        )
        plt.title("SHAP Decision Plot - Feature Contributions Path", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Decision plot saved to: {output_path}")
        plt.close()
        
    def waterfall_plot(self, instance_index=0, output_path=None, max_display=15):
        if self.shap_values is None:
            self.compute_shap_values()
        
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        
        explanation = shap.Explanation(
            values=shap_vals[instance_index],
            base_values=expected_value,
            data=self.X_data.iloc[instance_index].values,
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation, max_display=max_display, show=False)
        plt.title(f"SHAP Waterfall Plot - Instance {instance_index}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Waterfall plot saved to: {output_path}")
        plt.close()
        
    def get_feature_importance(self):
        if self.shap_values is None:
            self.compute_shap_values()
        
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        
        importance = np.abs(shap_vals).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        return importance_df
    
    def save_shap_values(self, output_path):
        if self.shap_values is None:
            self.compute_shap_values()
        
        joblib.dump({
            'shap_values': self.shap_values,
            'expected_value': self.explainer.expected_value,
            'feature_names': self.feature_names
        }, output_path)
        print(f"SHAP values saved to: {output_path}")
        
    def generate_all_plots(self, output_dir, instance_index=0, max_display=20):
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating SHAP Summary Plot...")
        self.summary_plot(
            output_path=os.path.join(output_dir, "shap_summary_plot.png"),
            max_display=max_display
        )
        
        print("Generating SHAP Bar Plot (Global Importance)...")
        self.bar_plot(
            output_path=os.path.join(output_dir, "shap_bar_plot.png"),
            max_display=max_display
        )
        
        print("Generating SHAP Force Plot (Local Explanation)...")
        self.force_plot(
            instance_index=instance_index,
            output_path=os.path.join(output_dir, f"shap_force_plot_instance_{instance_index}.png")
        )
        
        print("Generating SHAP Decision Plot...")
        self.decision_plot(
            instance_indices=list(range(min(10, len(self.X_data)))),
            output_path=os.path.join(output_dir, "shap_decision_plot.png"),
            max_display=max_display
        )
        
        print("Generating SHAP Waterfall Plot...")
        self.waterfall_plot(
            instance_index=instance_index,
            output_path=os.path.join(output_dir, f"shap_waterfall_plot_instance_{instance_index}.png"),
            max_display=max_display
        )
        
        importance_df = self.get_feature_importance()
        importance_path = os.path.join(output_dir, "feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        print(f"Feature importance saved to: {importance_path}")
        
        shap_values_path = os.path.join(output_dir, "shap_values.joblib")
        self.save_shap_values(shap_values_path)
        
        print(f"\nAll SHAP outputs generated successfully in: {output_dir}")
        return importance_df
