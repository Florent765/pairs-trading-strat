import os
import pandas as pd

RAW_PATH = "data/raw/prices.csv"
CLEAN_DIR = "data/clean"
OUT_NAME = "prices_clean.csv"

def clean_data(raw_path=RAW_PATH):

    os.makedirs(CLEAN_DIR, exist_ok=True)

    # Load

    df = pd.read_csv(raw_path, index_col=0, parse_dates=True)

    # Clean

    df = df.ffill().bfill()

    # Save

    out_path = os.path.join(CLEAN_DIR, OUT_NAME)
    df.to_csv(out_path)
    print(f"Clean data saved to {out_path}")
    return out_path

if __name__ == "__main__":
    clean_data()