# Pairs Trading Strategy with Machine Learning and Cointegration Filtering

This project implements a machine learning–driven **statistical arbitrage strategy** for trading **mean-reverting equity pairs**. The full pipeline includes pair selection via **cointegration tests**, feature engineering on spread dynamics, ML-based signal generation, and a robust backtesting framework with performance diagnostics.

> ✅ Designed for recruiters: This project demonstrates applied machine learning, financial modeling, and performance evaluation on real market data.

---

## 🔧 Tools & Technologies

- **Python**, **Pandas**, **NumPy**, **scikit-learn**, **statsmodels**
- **Random Forest Classifier** for binary prediction
- **Cointegration filtering** with ADF and CCM
- **Backtesting** with transaction costs, stop-loss, and Sharpe/drawdown metrics
- **Matplotlib**, **Seaborn** for equity curves and diagnostics

---

## 💡 Strategy Overview

1. **Pair Selection**
   - Select stock pairs from a broad universe (e.g. S&P 500)
   - Filter using **Augmented Dickey-Fuller (ADF)** and **Canonical Correlation Measure (CCM)** to find cointegrated pairs

2. **Spread Construction**
   - Fit OLS regression:  
     \[
     y_t = \alpha + \beta x_t \Rightarrow \text{spread}_t = y_t - (\alpha + \beta x_t)
     \]
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
   - Compare strategy to **SPY benchmark** and a **randomized baseline**

---

## 📊 Key Results

| Metric           | Value (Top Pairs)     |
|------------------|-----------------------|
| Sharpe Ratio     | Up to **2.5**         |
| Win/Loss Ratio   | > **1.2**             |
| Max Drawdown     | Controlled via stop-loss |
| Random vs ML     | ML significantly outperformed random baseline |
| Benchmark        | Strategy competitive vs SPY in volatile regimes |

---

## 📷 Notebook Highlights

- Equity curves: strategy vs SPY and randomized baseline  
- Top 5 & worst 3 pair performance  
- Portfolio drawdown and daily return histogram  
- Confusion matrix and ROC curves for model evaluation

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
