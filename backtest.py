import numpy as np
import pandas as pd

def compute_pnl(spread, signals, transaction_cost):
    spread_ret    = spread.diff().shift(-1)
    strat_ret     = signals * spread_ret

    trades = signals.diff().abs().fillna(0)
    tc_series = trades * transaction_cost
    strat_ret -= tc_series

    return strat_ret.cumsum(), strat_ret

def backtest_with_stop(spread, signals, stop_loss, transaction_cost):
    pnl = pd.Series(0.0, index=spread.index)
    position    = 0
    entry_price = 0.0

    for t in range(1, len(signals)):
        date = signals.index[t]
        sig  = signals.iat[t]
        prev = signals.iat[t-1]

        # Open new trade?
        if position == 0 and sig != 0:
            position    = sig
            entry_price = spread.iat[t]
            pnl.iat[t]  = -transaction_cost

        # If we have a trade open:
        elif position != 0:
            current_pl = position * (spread.iat[t] - entry_price)

            # Stop‐loss check
            if current_pl <= -stop_loss:
                pnl.iat[t]   = -stop_loss - transaction_cost
                position     = 0

            # Normal exit on signal flatten or reverse
            elif sig == 0 or sig == -position:
                pnl.iat[t]   = current_pl - transaction_cost
                position     = 0

            # else hold, no PnL booked today (we’ll book on exit)
    if position != 0:
        # book the final mark‐to‐market PnL at the last bar
        final_pl = position * (spread.iat[-1] - entry_price)
        pnl.iat[-1] = final_pl - transaction_cost
        position = 0

    return pnl.cumsum(), pnl

def evaluate(strategy_return):
    sharpe   = np.sqrt(252) * strategy_return.mean() / strategy_return.std()
    wins     = (strategy_return > 0).sum()
    losses   = (strategy_return < 0).sum()
    wl_ratio = wins / losses if losses > 0 else np.inf
    return sharpe, wl_ratio