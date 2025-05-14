import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier

def construct_spread(x, y):
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    alpha = model.params[0]
    beta = model.params[1]
    spread = y - (beta * x + alpha)
    return spread, beta, alpha

def compute_zscore(spread, window=60):
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread - mean) / std

def create_train_features(zscore, spread):
    df = {
        'zscore': zscore,
        'lag1': zscore.shift(1),
        'lag2': zscore.shift(2),
        'momentum': zscore - zscore.shift(5),
        'volatility': spread.rolling(10).std()
    }
    df = pd.DataFrame(df).dropna()
    df['future_z'] = zscore.shift(-1).loc[df.index]
    df['target'] = (abs(zscore.loc[df.index]) > abs(df['future_z'])).astype(int)
    return df.dropna()

def create_pred_features(spread_train, spread_test, window=60):
    spread_full = pd.concat([spread_train, spread_test])
    zfull = compute_zscore(spread_full, window=window)

    df = pd.DataFrame({
        'zscore':     zfull,
        'lag1':       zfull.shift(1),
        'lag2':       zfull.shift(2),
        'momentum':   zfull - zfull.shift(5),
        'volatility': spread_full.rolling(10).std(),
    })

    df_test = df.loc[spread_test.index].dropna()
    return df_test

def train_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

def generate_signals(zscore, predicted):
    def decision(p, z):
        if p == 1:
            if z > 1:   return -1
            if z < -1:  return 1
        return 0
    return zscore.combine(predicted, func=lambda z, p: decision(p, z))