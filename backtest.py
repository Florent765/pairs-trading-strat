import numpy as np

def compute_pnl(spread, signals):
    spread_ret    = spread.diff().shift(-1)
    strat_ret     = signals * spread_ret
    return strat_ret.cumsum(), strat_ret

def evaluate(strategy_return):
    sharpe   = np.sqrt(252) * strategy_return.mean() / strategy_return.std()
    wins     = (strategy_return > 0).sum()
    losses   = (strategy_return < 0).sum()
    wl_ratio = wins / losses if losses > 0 else np.inf
    return sharpe, wl_ratio