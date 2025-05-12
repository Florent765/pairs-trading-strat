import os
import yfinance as yf

RAW_DIR = "data/raw"

def fetch_data(tickers, start_date, end_date):
    # Download historical data
    os.makedirs(RAW_DIR, exist_ok=True)

    df = yf.download(tickers, start=start_date, end=end_date)["Close"]

    out_path = os.path.join(RAW_DIR, "prices.csv")
    df.to_csv(out_path, index=False)
    print(f"Raw data saved to {out_path}")
    return out_path

if __name__ == "__main__":
    # Example invocation
    fetch_data(["KO", "PEP"], "2013-01-02", "2021-09-28")