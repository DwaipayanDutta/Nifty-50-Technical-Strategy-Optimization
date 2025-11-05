# =========================================================
# 📈 Nifty50 Strategy Optimization (2000–2021)
# =========================================================
# --- Setup ---
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-darkgrid")

# =========================================================
# 1️⃣ Load Data
# =========================================================
def load_combined_csv(filepath: str) -> pd.DataFrame:
    """
    Loads combined Nifty50 CSV containing multiple stocks.
    Expected columns: Date, Stock, Open, High, Low, Close, Volume
    """
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Stock', 'Date'])
    return df

# Provide your dataset path here
filepath = r"combined_nifty50_data.csv"

df = load_combined_csv(filepath)
print("✅ Data Loaded Successfully")
display(df.head())

# =========================================================
# 2️⃣ Compute Technical Indicators
# =========================================================
def compute_indicators(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    df = df.copy()
    df = df.set_index('Date')

    # Moving averages
    df['ma_short'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
    df['ma_long'] = df['Close'].rolling(window=long_window, min_periods=1).mean()

    # Momentum (5-day)
    df['momentum_5'] = df['Close'].pct_change(periods=5, fill_method=None)

    # RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Volatility (20-day std of returns)
    df['vol_20'] = df['Close'].pct_change(fill_method=None).rolling(window=20, min_periods=1).std()

    return df.dropna()

# =========================================================
# 3️⃣ Signal Generation
# =========================================================
def generate_signals(df: pd.DataFrame, rsi_low: float, rsi_high: float) -> pd.DataFrame:
    """
    Generate trading signals using MA crossover and RSI filter.
    """
    df = df.copy()
    df['signal'] = 0
    df['position'] = 0

    df['ma_diff'] = df['ma_short'] - df['ma_long']
    df['crossover_up'] = (df['ma_diff'] > 0) & (df['ma_diff'].shift(1) <= 0)
    df['crossover_down'] = (df['ma_diff'] < 0) & (df['ma_diff'].shift(1) >= 0)

    for idx in df.index:
        if df.at[idx, 'crossover_up'] and df.at[idx, 'rsi'] < rsi_low:
            df.at[idx, 'signal'] = 1
        elif df.at[idx, 'crossover_down'] or df.at[idx, 'rsi'] > rsi_high:
            df.at[idx, 'signal'] = -1

    # Position logic
    df['position'] = 0
    position = 0
    for idx in df.index:
        if df.at[idx, 'signal'] == 1:
            position = 1
        elif df.at[idx, 'signal'] == -1:
            position = 0
        df.at[idx, 'position'] = position

    return df

# =========================================================
# 4️⃣ Backtesting
# =========================================================
def backtest(df: pd.DataFrame, initial_capital=100000, fee=0.0005):
    df = df.copy()
    df['returns'] = df['Close'].pct_change().fillna(0)
    df['strategy_returns'] = df['position'].shift(1) * df['returns'] - fee * abs(df['position'].diff())
    df['equity_curve'] = (1 + df['strategy_returns']).cumprod() * initial_capital

    total_return = df['equity_curve'].iloc[-1] / initial_capital - 1
    annualised_return = (1 + total_return) ** (252 / len(df)) - 1
    annualised_vol = df['strategy_returns'].std() * np.sqrt(252)
    sharpe_ratio = annualised_return / annualised_vol if annualised_vol != 0 else np.nan

    rolling_max = df['equity_curve'].cummax()
    drawdown = (df['equity_curve'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        'total_return': total_return,
        'annualised_return': annualised_return,
        'annualised_vol': annualised_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'equity_curve': df['equity_curve'],
        'df': df
    }

# =========================================================
# 5️⃣ Optimization Loop
# =========================================================
def optimise_parameters(df, short_windows, long_windows, rsi_lows, rsi_highs):
    results = []
    stocks = df['Stock'].unique()

    for stock in stocks:
        df_stock = df[df['Stock'] == stock].copy()
        df_stock = df_stock.sort_values('Date').reset_index(drop=True)

        best_sharpe = -np.inf
        best_params = None

        for short_w, long_w, rsi_low, rsi_high in itertools.product(short_windows, long_windows, rsi_lows, rsi_highs):
            if short_w >= long_w:
                continue

            try:
                df_ind = compute_indicators(df_stock, short_w, long_w)
                df_sig = generate_signals(df_ind, rsi_low, rsi_high)
                metrics = backtest(df_sig)
                if np.isnan(metrics['sharpe_ratio']):
                    continue

                if metrics['sharpe_ratio'] > best_sharpe:
                    best_sharpe = metrics['sharpe_ratio']
                    best_params = {
                        'Stock': stock,
                        'short_window': short_w,
                        'long_window': long_w,
                        'rsi_low': rsi_low,
                        'rsi_high': rsi_high,
                        'sharpe_ratio': metrics['sharpe_ratio'],
                        'total_return': metrics['total_return'],
                        'max_drawdown': metrics['max_drawdown']
                    }
            except Exception as e:
                print(f"⚠️ Error processing {stock} ({short_w}, {long_w}, {rsi_low}, {rsi_high}): {str(e)}")
                continue

        if best_params:
            results.append(best_params)
        else:
            print(f"🚫 No valid strategy for {stock}")

    if not results:
        print("🚫 No valid results found for any stock")
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='sharpe_ratio', ascending=False).reset_index(drop=True)
    return df_results

# =========================================================
# 6️⃣ Run Optimization
# =========================================================
short_windows = [20, 50, 100]
long_windows = [100, 150, 200]
rsi_lows = [30, 40]
rsi_highs = [70, 80]

print("🚀 Running parameter optimization across Nifty50 stocks...")
optimized_results = optimise_parameters(df, short_windows, long_windows, rsi_lows, rsi_highs)

print("\n✅ Top 10 Optimized Strategies:")
display(optimized_results.head(10))

# =========================================================
# 7️⃣ Visualization: Sharpe Ratios
# =========================================================
plt.figure(figsize=(12, 6))
sns.barplot(x='Stock', y='sharpe_ratio', data=optimized_results.head(15), palette="coolwarm")
plt.title("Top 15 Stocks by Sharpe Ratio (Optimized Strategy)")
plt.xticks(rotation=45)
plt.show()

# =========================================================
# 8️⃣ Example Equity Curve Visualization
# =========================================================
top_stock = optimized_results.iloc[0]['Stock']
top_short = int(optimized_results.iloc[0]['short_window'])
top_long = int(optimized_results.iloc[0]['long_window'])
top_rsi_low = float(optimized_results.iloc[0]['rsi_low'])
top_rsi_high = float(optimized_results.iloc[0]['rsi_high'])

df_top = df[df['Stock'] == top_stock].copy()
df_ind = compute_indicators(df_top, top_short, top_long)
df_sig = generate_signals(df_ind, top_rsi_low, top_rsi_high)
metrics = backtest(df_sig)

plt.figure(figsize=(10, 5))
plt.plot(metrics['equity_curve'], label="Equity Curve", color='blue')
plt.title(f"Equity Curve – {top_stock} (Optimized Params)")
plt.legend()
plt.show()

print(f"📊 {top_stock} – Sharpe: {metrics['sharpe_ratio']:.2f}, Total Return: {metrics['total_return']*100:.1f}%")
