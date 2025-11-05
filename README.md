# 📈 Nifty 50 Trading Strategy Optimization (2000–2021)

## 🧭 Overview

This project builds and optimizes a **systematic trading strategy** across **Nifty 50 stocks (2000–2021)** using historical market data.  
It uses **technical indicators** (Moving Averages, RSI, Momentum, and Volatility) to generate buy/sell signals and optimizes parameters using a **Sharpe ratio–based grid search**.

The complete process — from data ingestion to optimization and visualization — is implemented in a **Jupyter Notebook** for transparency and reproducibility.

---

## ⚙️ Features

- Loads and processes Nifty 50 stock data (2000–2021)
- Computes multiple technical indicators
- Generates trading signals using MA crossover + RSI filters
- Backtests strategies to calculate returns, Sharpe ratio, and drawdowns
- Performs parameter optimization across all stocks
- Visualizes performance and top-optimized strategies
- Outputs a ranked list of stocks by Sharpe ratio

---

## 📁 Project Structure

Nifty50_Strategy_Optimization/  
│  
├── Nifty50_Strategy_Optimization.ipynb         # Main Jupyter notebook implementing the strategy  
├── combined_nifty50_data.csv                   # Combined stock data CSV file (source: Kaggle)  
└── README.md                                   # Documentation and project overview  

---
## 📊 Approach Note

### 🎯 Objective

Develop a **quantitative trading strategy** to identify profitable buy/sell opportunities across the Nifty 50 universe.  
The strategy combines **trend-following** (via moving averages) and **momentum** (via RSI and returns) techniques and aims to **maximize risk-adjusted returns**.

---

### 🧩 1. Data Handling

- Dataset: `combined_nifty50_data.csv` (contains all Nifty 50 tickers)
- Columns: `Date`, `Stock`, `Open`, `High`, `Low`, `Close`, `Volume`
- The notebook converts `Date` to datetime, sorts chronologically, and processes data per stock.

---

### 📈 2. Technical Indicators

| Indicator | Description | Purpose |
|------------|--------------|----------|
| **MA (short)** | Moving average (short-term) | Captures short-term trend |
| **MA (long)** | Moving average (long-term) | Captures long-term trend |
| **RSI (14)** | Relative Strength Index | Detects overbought/oversold zones |
| **Momentum (5)** | 5-day % change | Measures recent acceleration |
| **Volatility (20)** | Rolling std of returns | Quantifies price risk |

Computation is vectorized using pandas for speed and accuracy.

---

### ⚙️ 3. Strategy Logic

#### Entry (Buy):
- Short-term MA crosses **above** long-term MA (trend confirmation)
- RSI is **below RSI Low** threshold (not overbought)

#### Exit (Sell):
- Short-term MA crosses **below** long-term MA
- RSI is **above RSI High** threshold (overbought)

Positions are held until the next sell condition is met.

---

### 💰 4. Backtesting Framework

For each stock and parameter combination:
1. Generate trading signals  
2. Simulate position changes over time  
3. Calculate daily returns and cumulative equity  
4. Derive metrics:
   - **Total Return**
   - **Annualized Return**
   - **Sharpe Ratio**
   - **Max Drawdown**

Formula:
```python
strategy_returns = position.shift(1) * returns - fee * abs(position.diff())
equity_curve = (1 + strategy_returns).cumprod()




