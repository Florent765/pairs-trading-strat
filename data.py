import os
import yfinance as yf
from datetime import datetime, timedelta

RAW_DIR = "data/raw"

sp500_tech = [
    "MSFT", "AAPL", "NVDA", "AVGO", "ORCL", "PLTR", "CRM", "CSCO", "IBM",
    "NOW", "ACN", "INTU", "AMD", "TXN", "ADBE", "QCOM", "AMAT", "PANW",
    "ANET", "ADI", "CRWD", "LRCX", "MU", "APH", "KLAC", "INTC", "CDNS",
    "SNPS", "FTNT", "DELL", "WDAY", "MSI", "ADSK", "ROP", "NXPI", "FICO",
    "TEL", "CTSH", "GLW", "IT", "MPWR", "MCHP", "ANSS", "KEYS", "GDDY",
    "HPQ", "VRSN"
]

def get_train_test_dates():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # Last 6 months
    train_test_split = end_date - timedelta(days=30)  # Last month for testing
    
    return start_date.strftime('%Y-%m-%d'), train_test_split.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def fetch_data(tickers, start_date, end_date):
    # Download historical data
    os.makedirs(RAW_DIR, exist_ok=True)

    df = yf.download(tickers, start=start_date, end=end_date)["Close"]

    out_path = os.path.join(RAW_DIR, "prices.csv")
    df.to_csv(out_path, index=False)
    print(f"Raw data saved to {out_path}")
    return out_path

if __name__ == "__main__":
    start_date, split_date, end_date = get_train_test_dates()
    fetch_data(sp500_tech, start_date, end_date)