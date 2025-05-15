import os
import pandas as pd
from joblib import load
from model import compute_zscore, create_pred_features, generate_signals, construct_spread
from backtest import compute_pnl, evaluate

os.makedirs("results", exist_ok=True)

# 1. Load test data & same pairs list
prices_test = pd.read_csv("data/clean/test_prices_clean.csv", index_col=0, parse_dates=True)
prices_train = pd.read_csv("data/clean/train_prices_clean.csv", index_col=0, parse_dates=True)

pairs = pd.read_csv("data/processed/universal_pairs.csv")

results = []
for _, row in pairs.iterrows():
    A, B = row['stock1'], row['stock2']
    print(f"Backtesting {A}/{B}")

    # Reconstruct the train‐period spread
    price_A_tr = prices_train[A]
    price_B_tr = prices_train[B]
    spread_tr, beta, alpha = construct_spread(price_B_tr, price_A_tr)

    # Reconstruct spread on test set
    price_A_te = prices_test[A]
    price_B_te = prices_test[B]
    spread_te = price_A_te - (alpha + beta * price_B_te)

    # Build prediction features
    X_te = create_pred_features(
        spread_train=spread_tr,
        spread_test=spread_te,
        window=60
    )[['lag1', 'lag2', 'momentum', 'volatility']]

    # Load the trained model + α/β
    model, alpha, beta = load(f"models/{A}_{B}.joblib")

    # 5. Predict & generate signals
    y_pred = model.predict(X_te)
    signals = generate_signals(
        zscore=compute_zscore(pd.concat([spread_tr, spread_te])).loc[X_te.index],
        predicted=pd.Series(y_pred, index=X_te.index)
    )

    # 6. Backtest
    pnl_curve, daily_ret = compute_pnl(spread_te.loc[X_te.index], signals)
    sharpe, wl_ratio = evaluate(daily_ret)
    trades = signals.abs().sum()

    results.append({
        'pair': f"{A}-{B}",
        'sharpe': round(sharpe, 3),
        'wl_ratio': round(wl_ratio, 2),
        'num_trades': int(trades)
    })

    # Optionally: save each PnL curve or signals
    pnl_curve.to_csv(f"results/pnl-{A}-{B}.csv")
    print(f" → {A}/{B}: Sharpe={sharpe:.2f}, W/L={wl_ratio:.2f}")

# 7. Summarize all pairs
summary = pd.DataFrame(results)
summary.to_csv("results/backtest_summary.csv", index=False)
print("\nTop pairs by Sharpe:\n", summary.sort_values('sharpe', ascending=False).head())