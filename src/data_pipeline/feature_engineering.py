import pandas as pd
import numpy as np
import os

def load_data(filepath):
    return pd.read_csv(filepath)

def create_physics_features(df):
    """
    Creates features based on physical properties and relationships.
    """
    df_feat = df.copy()
    
    # Temperature difference: Process - Air
    df_feat['Temperature_difference [K]'] = df_feat['Process temperature [K]'] - df_feat['Air temperature [K]']
    
    # Power: P = 2 * pi * n * T / 60
    # where n is rotational speed in rpm, T is torque in Nm. Result in Watts.
    df_feat['Power [W]'] = 2 * np.pi * df_feat['Rotational speed [rpm]'] * df_feat['Torque [Nm]'] / 60
    
    # Strain / Load proxy: Torque * Tool wear
    # High torque with high wear might indicate impending failure.
    df_feat['Wear_Torque_Interaction'] = df_feat['Torque [Nm]'] * df_feat['Tool wear [min]']
    
    return df_feat

def create_rolling_features(df, windows=[3, 5]):
    """
    Creates rolling mean and standard deviation features.
    Assumes data is sequential (time-series).
    """
    df_feat = df.copy()
    cols_to_roll = ['Air temperature [K]', 'Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]']
    
    for window in windows:
        for col in cols_to_roll:
            # Rolling Mean
            df_feat[f'{col}_rolling_mean_{window}'] = df_feat[col].rolling(window=window).mean()
            # Rolling Std
            df_feat[f'{col}_rolling_std_{window}'] = df_feat[col].rolling(window=window).std()
            
    # Rolling features introduce NaNs at the beginning, we can fill them with the first valid value or drop.
    # Forward fill then backward fill to handle initial NaNs
    df_feat.fillna(method='bfill', inplace=True)
    
    return df_feat

def encode_categorical(df):
    """
    Encodes categorical variables.
    """
    df_feat = df.copy()
    # 'Type' is categorical (L, M, H). Map to ordinal integers.
    type_map = {'L': 1, 'M': 2, 'H': 3}
    df_feat['Type_Ordinal'] = df_feat['Type'].map(type_map)
    df_feat['Type_Ordinal'] = df_feat['Type_Ordinal'].fillna(1) 
    
    return df_feat

def main():
    input_path = "data/processed/cleaned_sensor_data.csv"
    output_path = "data/processed/feature_engineered_data.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please ensure preprocess.py has executed successfully.")
        return

    print("Loading data...")
    df = load_data(input_path)
    
    print("Creating physics-based features...")
    df = create_physics_features(df)
    
    print("Creating rolling features...")
    df = create_rolling_features(df)
    
    print("Encoding categorical variables...")
    df = encode_categorical(df)
    
    print(f"Saving feature engineered data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Feature engineering completed successfully.")

if __name__ == "__main__":
    main()
