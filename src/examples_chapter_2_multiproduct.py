import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multi_product as mp

np.random.seed(42)

# --- Synthetic Data Setup ---
# We simulate 3 instruments over 200 bars
n_bars = 200
n_instruments = 3
instruments = ['A', 'B', 'C']
bars = pd.date_range(start='2020-01-01', periods=n_bars)

# Simulate raw open and close prices for each instrument
close_prices = pd.DataFrame(
    np.cumsum(np.random.randn(n_bars, n_instruments), axis=0) + 100,
    index=bars, columns=instruments
)
open_prices = close_prices + np.random.randn(n_bars, n_instruments) * 0.1

# Point values (USD value of one point) — set to 1 for simplicity
point_values = pd.DataFrame(
    np.ones((n_bars, n_instruments)),
    index=bars, columns=instruments
)

# Dividends — set to 0 for simplicity
dividends = pd.DataFrame(
    np.zeros((n_bars, n_instruments)),
    index=bars, columns=instruments
)

# Allocation weights — equal weight across all instruments
alloc_weights = pd.DataFrame(
    np.ones((n_bars, n_instruments)) / n_instruments,
    index=bars, columns=instruments
)

# Transaction costs — 1 basis point per instrument
trans_costs = pd.Series(1e-4, index=instruments)

# Rebalance every 20 bars
rebalance_dates = list(bars[::20])

# --- Section 2.4.2: PCA Weights ---
print("--- PCA Weights ---")
# Compute covariance matrix of returns
returns = close_prices.pct_change().dropna()
cov = returns.cov().values

# Equal risk distribution across all components
risk_dist = np.ones(n_instruments) / n_instruments
weights = mp.pca_weights(cov, risk_dist=risk_dist, risk_target=1.0)
print("PCA Weights:")
for i, w in zip(instruments, weights.flatten()):
    print(f"  {i}: {w:.4f}")

# --- Section 2.4.1: ETF Trick ---
print("\n--- ETF Trick ---")
result = mp.etf_trick(
    open_prices=open_prices,
    close_prices=close_prices,
    alloc_weights=alloc_weights,
    point_values=point_values,
    dividends=dividends,
    rebalance_dates=rebalance_dates,
    trans_costs=trans_costs
)
print(result.head(10))

# --- Section 2.4.3: Single Future Roll ---
print("\n--- Single Future Roll ---")

# Simulate a single futures series with 3 contract rolls
n = 200
roll_every = 67  # roll approximately every 67 bars

# Create synthetic ticker that changes at each roll
tickers = ['ESH20'] * roll_every + ['ESM20'] * roll_every + ['ESU20'] * (n - 2 * roll_every)
prices = np.cumsum(np.random.randn(n)) + 100

# Add a jump at each roll to simulate the contract price gap
prices[roll_every] += 2.0
prices[roll_every * 2] += -1.5

futures_df = pd.DataFrame({
    'Instrument': tickers,
    'Open': prices + np.random.randn(n) * 0.1,
    'Close': prices
}, index=pd.date_range(start='2020-01-01', periods=n))

# Apply roll gap correction
dictio = {'Instrument': 'Instrument', 'Open': 'Open', 'Close': 'Close'}

rolled = mp.get_rolled_series(futures_df, dictio=dictio, match_end=True)
non_neg = mp.non_negative_rolled_prices(futures_df, dictio=dictio, match_end=True)

print("Rolled series (first 10 rows):")
print(rolled.head(10))
print("\nNon-negative rolled prices (first 10 rows):")
print(non_neg[['Close', 'Returns', 'rPrices']].head(10))

# --- Plots ---
fig, axes = plt.subplots(4, 1, figsize=(12, 18))
fig.suptitle("Chapter 2.4 — Dealing with Multi-Product Series", fontsize=14, y=0.98)
# Plot 1: ETF Trick portfolio value
axes[0].plot(result.index, result['K'], color='blue')
axes[0].set_title("ETF Trick: Portfolio Value of $1 Invested Across 3 Instruments (Section 2.4.1)")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Portfolio Value ($)")

# Plot 2: Raw vs rolled futures prices
axes[1].plot(futures_df.index, futures_df['Close'], color='red', label='Raw prices')
axes[1].plot(rolled.index, rolled['Close'], color='green', label='Rolled prices')
axes[1].set_title("Single Future Roll: Removing Artificial Price Jumps at Contract Expiry (Section 2.4.3)")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Price")
axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
axes[1].set_ylim(90, 110)

# Plot 3: Non-negative rolled price series
axes[2].plot(non_neg.index, non_neg['rPrices'], color='purple')
axes[2].set_title("Single Future Roll: $1 Investment Value After Roll Correction (Section 2.4.3)")
axes[2].set_xlabel("Date")
axes[2].set_ylabel("Value ($)")

# Plot 4: PCA Weights
weight_values = weights.flatten()
colors = ['green' if w > 0 else 'red' for w in weight_values]
axes[3].bar(instruments, weight_values, color=colors)
axes[3].axhline(y=0, color='black', linewidth=0.8)
axes[3].set_title("PCA Weights: Optimal Risk-Balanced Allocation Across Instruments (Section 2.4.2)")
axes[3].set_xlabel("Instrument")
axes[3].set_ylabel("Weight")
for i, w in enumerate(weight_values):
    y_pos = w / 2  # center of the bar
    axes[3].text(i, y_pos, f'{w:.2f}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

plt.subplots_adjust(hspace=0.8, right=0.85)
plt.show()