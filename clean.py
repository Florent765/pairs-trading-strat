import os
import pandas as pd

RAW_PATH = "data/raw/prices.csv"
CLEAN_DIR = "data/clean"
OUT_NAME = "prices_clean.csv"

def split_train_test(df):
    train_end = df.index[-90]  # Last 30 days for testing
    
    train_data = df[df.index < train_end]
    test_data = df[df.index >= train_end]
    
    return train_data, test_data

def clean_data(raw_path=RAW_PATH):

    os.makedirs(CLEAN_DIR, exist_ok=True)

    # Load
    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)

    # Clean
    df = df.ffill().bfill()

    # Split
    train_data, test_data = split_train_test(df)

    # Save
    train_path = os.path.join(CLEAN_DIR, "train_" + OUT_NAME)
    test_path = os.path.join(CLEAN_DIR, "test_" + OUT_NAME)
    
    train_data.to_csv(train_path)
    test_data.to_csv(test_path)
    
    print(f"Training data saved to {train_path}")
    print(f"Testing data saved to {test_path}")
    
    return train_path, test_path

if __name__ == "__main__":
    clean_data()