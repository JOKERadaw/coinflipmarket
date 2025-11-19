import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# Force matplotlib to not use any Xwindow/Tkinter backend
matplotlib.use('Agg')
# ==========================================
# 1. Configuration & Data
# ==========================================
TICKER = "NVDA"
START = "2020-01-01"
END = "2024-01-01"
NUM_SIMULATIONS = 10000  # "Large amount of temptatives"

print(f"Fetching data for {TICKER}...")
data = yf.download(TICKER, start=START, end=END, progress=False)

# USER REQUEST: Use 'Close' explicitly
# Depending on yfinance version, this might return a Series or DataFrame. 
# We ensure it's a clean Series.
if 'Close' in data.columns:
    prices = data['Close']
else:
    # Fallback if columns are multi-level (common in new yfinance)
    prices = data.iloc[:, 0] 

# Calculate Daily Percentage Returns
# Fill NaN with 0 to prevent math errors in matrix multiplication
market_returns = prices.pct_change().fillna(0).to_numpy()

# ==========================================
# 2. Vectorized Random Simulation
# ==========================================
print(f"Running {NUM_SIMULATIONS} random simulations...")

# Time Horizon (number of trading days)
T = len(market_returns)

# Create a massive Matrix of Random Choices
# 0 = Cash (Out of market), 1 = Invested (In market)
# Shape: (Days, Simulations)
# p=[0.5, 0.5] means 50% chance to be in or out on any given day
decision_matrix = np.random.choice([0, 1], size=(T, NUM_SIMULATIONS), p=[0.5, 0.5])

# Broadcast multiplication:
# Multiply Market Returns (Column Vector) by Decision Matrix
# If decision is 0, return is 0. If decision is 1, return is market return.
simulated_daily_returns = market_returns.reshape(-1, 1) * decision_matrix

# Calculate Cumulative Returns for all 10,000 simulations simultaneously
# (1 + r) * (1 + r)...
equity_curves = np.cumprod(1 + simulated_daily_returns, axis=0)

# Calculate Buy & Hold (Benchmark)
benchmark_curve = np.cumprod(1 + market_returns)

# ==========================================
# 3. Analysis & Visualization
# ==========================================
final_values = equity_curves[-1, :] # The last row contains final results
benchmark_final = benchmark_curve[-1]

print("\n--- ANALYSIS OF RANDOMNESS ---")
print(f"Asset: {TICKER} (Buy & Hold Total Return: {(benchmark_final-1)*100:.2f}%)")
print(f"Simulations: {NUM_SIMULATIONS}")
print("-" * 30)
print(f"Average Random Return: {(np.mean(final_values)-1)*100:.2f}%")
print(f"Best Random Run:      {(np.max(final_values)-1)*100:.2f}%")
print(f"Worst Random Run:     {(np.min(final_values)-1)*100:.2f}%")

# Probability of beating Buy & Hold
# We count how many random paths ended higher than the benchmark
beat_benchmark = np.sum(final_values > benchmark_final)
prob_success = (beat_benchmark / NUM_SIMULATIONS) * 100

print("-" * 30)
print(f"Probability of beating Buy & Hold randomly: {prob_success:.2f}%")

# --- PLOTTING ---
plt.figure(figsize=(12, 6))

# 1. Histogram of Outcomes
plt.subplot(1, 2, 1)
# Use Log scale for x-axis if volatility is huge, otherwise linear
plt.hist(final_values, bins=100, color='teal', alpha=0.7, label='Random Strategies')
plt.axvline(benchmark_final, color='red', linestyle='dashed', linewidth=2, label='Buy & Hold')
plt.axvline(np.mean(final_values), color='yellow', linestyle='solid', linewidth=2, label='Mean Random')
plt.title(f"Distribution of {NUM_SIMULATIONS} Random Outcomes")
plt.xlabel("Final Multiplier (1.0 = Break Even)")
plt.ylabel("Frequency")
plt.legend()

# 2. Spaghetti Plot (First 100 paths only, to keep it readable)
plt.subplot(1, 2, 2)
plt.plot(equity_curves[:, :100], color='gray', alpha=0.1)
plt.plot(benchmark_curve, color='red', linewidth=2, label='Buy & Hold')
plt.title(f"Trajectories (Showing 100 of {NUM_SIMULATIONS})")
plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.legend()

plt.tight_layout()
plt.savefig("ciao.png")