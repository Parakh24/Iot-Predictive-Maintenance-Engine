import pandas as pd

# Load feature-engineered data
df = pd.read_csv("data/processed/feature_engineered_data.csv") 

# Select only numeric columns
numeric_df = df.select_dtypes(include='number')

# Compute correlation between all numeric features
corr = numeric_df.corr() 

# Save correlation matrix as CSV
corr.to_csv("data/processed/correlation_matrix.csv") 

print("Correlation matrix saved to data/processed/correlation_matrix.csv")
