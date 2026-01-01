import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "week2")
csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")

# Load CSV
df = pd.read_csv(csv_path)

# ------------------------
# 1. Print simple summary
# ------------------------
print("\nModel Comparison Report:")
print(df.sort_values("F1", ascending=False))

# ------------------------
# 2. Visualize comparison
# ------------------------
plt.figure(figsize=(8,5))
plt.bar(df["Model"], df["F1"], color="skyblue", alpha=0.8, label="F1 Score")
plt.bar(df["Model"], df["Recall"], color="orange", alpha=0.5, label="Recall")
plt.title("Model Comparison: F1 vs Recall")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save plot
plot_path = os.path.join(RESULTS_DIR, "model_comparison_plot.png")
plt.savefig(plot_path)
plt.show()
print(f"\nComparison plot saved at: {plot_path}")
