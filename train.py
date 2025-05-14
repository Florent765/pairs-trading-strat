import pandas as pd
from joblib import dump
from model import (
    construct_spread,
    compute_zscore,
    create_train_features,
    train_model
)

# Load training data & selected pairs
prices = pd.read_csv("data/clean/train_prices_clean.csv", index_col=0, parse_dates=True)
pairs = pd.read_csv("data/processed/universal_pairs.csv")  # columns: Stock1, Stock2

# Loop over each pair, train a model, save model + α/β
for _, row in pairs.iterrows():
    A, B = row['stock1'], row['stock2']
    print(f"Training on {A}/{B}")

    price_A = prices[A]
    price_B = prices[B]

    # Spread & z-score
    spread, beta, alpha = construct_spread(price_B, price_A)
    zscore = compute_zscore(spread)

    # 4. Build training features & target
    df_tr = create_train_features(zscore, spread)
    X_tr = df_tr[['lag1', 'lag2', 'momentum', 'volatility']]
    y_tr = df_tr['target']

    # 5. Train and save
    model = train_model(X_tr, y_tr)
    dump((model, alpha, beta), f"models/{A}_{B}.joblib")
    print(f" → Saved model and α/β to models/{A}_{B}.joblib")