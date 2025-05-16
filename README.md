# Pairs Trading Strategy with Machine Learning and Cointegration Filtering


This project implements a machine learning–driven **statistical arbitrage strategy** for trading **mean-reverting equity pairs**. The full pipeline includes cointegration testing, nonlinear dependency filtering, ML-based signal generation, and a backtesting engine with risk controls.

---

## Tools & Technologies

- **Python**, **Pandas**, **NumPy**, **scikit-learn**, **statsmodels**
- **Random Forest Classifier** for reversion prediction
- **Cointegration** filtering (Engle-Granger + ADF) + **CCM-KNN** skill score
- **Backtesting** with transaction costs, stop-loss, and Sharpe/drawdown metrics
- **Matplotlib**, **Seaborn** for equity curves and diagnostics

---

## Strategy Overview

1. **Pair Selection**
   - Select stock pairs from a broad universe (e.g. S&P 500)
   - Apply **Engle-Granger test** with **ADF** to identify cointegrated pairs
   - Refine selection using **Convergent Cross Mapping (CCM)** and **K-Nearest Neighbors (KNN)** to retain only the most dynamically coupled pairs
     
2. **Spread Construction**
   - Fit OLS regression:  
     y_t = α + β x_t → spread_t = y_t − (α + β x_t)
   - Normalize the spread using a rolling 60-day **z-score**

3. **Feature Engineering**
   - Lagged z-scores, momentum, volatility over rolling windows
   - Target: predict whether the spread will **mean-revert** the next day

4. **Modeling**
   - Train a **Random Forest** classifier on historical spread behavior
   - Generate trading signals based on model prediction and z-score thresholds

5. **Backtesting**
   - Execute trades with transaction costs and optional stop-loss
   - Evaluate:
     - **Sharpe ratio**
     - **Win/loss ratio**
     - **Max drawdown**
     - **Daily return distribution**
   - Compare strategy to **SPY benchmark**

---

## Key Results

| Metric           | Value (Top Pairs)     |
|------------------|-----------------------|
| Sharpe Ratio     | Up to **1.7**         |
| Win/Loss Ratio   | > **1.2**             |
| Max Drawdown     | Controlled via stop-loss |
| Benchmark        | Strategy competitive vs SPY in volatile regimes |

---

## How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
