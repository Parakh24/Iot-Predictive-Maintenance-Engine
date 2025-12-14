import pandas as pd
import os

def load_data():
    df = pd.read_csv("data/raw/Predictive Maintainance dataset.csv")
    return df

def handle_missing_values(df):
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    return df

def remove_corrupted_values(df):
    df = df[df["Air temperature [K]"] > 0]
    df = df[df["Process temperature [K]"] > 0]
    df = df[df["Rotational speed [rpm]"] >= 0]
    df = df[df["Torque [Nm]"] >= 0]
    df = df[df["Tool wear [min]"] >= 0]
    return df

def main():
    os.makedirs("data/processed", exist_ok=True)

    df = load_data()
    df = handle_missing_values(df)
    df = remove_corrupted_values(df)

    df.to_csv("data/processed/cleaned_sensor_data.csv", index=False)
    print("Data cleaning completed successfully.")

if __name__ == "__main__":
    main()
