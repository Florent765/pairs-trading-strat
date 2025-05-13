import itertools
import os
import itertools as it
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

CLEAN_PATH = "data/clean/prices_clean.csv"
OUT_DIR = "data/processed"
OUT_FILE = "scanned_pairs.csv"

def eagle_granger_test(x, y, p=0.05):
    # OLS regression
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit().resid

    # ADF test
    pvalue = adfuller(res, regression="c")[1]
    return pvalue < p, pvalue

def scan_pairs(df, p=0.05):
    tickers = df.columns.tolist()
    res = []

    for t1, t2 in itertools.combinations(tickers, 2):
        x = df[t1]
        y = df[t2]
        coint, pvalue = eagle_granger_test(x, y, p)
        if coint:
            res.append((t1, t2, pvalue))
    return res

if __name__ == '__main__':

    # Read from data
    df = pd.read_csv(CLEAN_PATH, index_col=0, parse_dates=True)

    # Scan
    pairs = scan_pairs(df, p=0.05)
    if not pairs:
        print("No pairs found at p < 0.05")
    else:
        os.makedirs(OUT_DIR, exist_ok=True)
        out_df = pd.DataFrame(pairs, columns=["stock1","stock2","pvalue"])
        out_df.to_csv(os.path.join(OUT_DIR, OUT_FILE), index=False)
        print(f"Found {len(pairs)} pairs, saved to {OUT_DIR}/{OUT_FILE}")
        print(out_df)
