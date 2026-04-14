import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multi_product as mp

# --- Load SP Futures Data ---
base = os.path.dirname(os.path.abspath(__file__))

def load_contract(filename):
    path = os.path.join(base, 'input_data', filename)
    # Read first row to check if it's a header or data
    first = pd.read_csv(path, nrows=1, header=None)
    has_header = not str(first.iloc[0, 0]).isdigit()
    
    if has_header:
        df = pd.read_csv(path)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt']
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        df = pd.read_csv(path, header=None,
                         names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt'])
        df['Date'] = pd.to_datetime(df['Date'], format='%y%m%d')
    return df

# --- Section 2.4.2 & 2.4.1: ETF Trick + PCA Weights ---
# Use three contracts as a basket of instruments
print("Loading contracts for ETF Trick and PCA Weights...")
sp98h = load_contract('SP98H.txt')
sp98m = load_contract('SP98M.txt')
sp98u = load_contract('SP98U.txt')

# Align dates across all three contracts
dates = sp98h['Date']
common_dates = dates[dates.isin(sp98m['Date']) & dates.isin(sp98u['Date'])]

sp98h_aligned = sp98h[sp98h['Date'].isin(common_dates)].set_index('Date')
sp98m_aligned = sp98m[sp98m['Date'].isin(common_dates)].set_index('Date')
sp98u_aligned = sp98u[sp98u['Date'].isin(common_dates)].set_index('Date')

# Build multi-instrument dataframes
instruments = ['SP98H', 'SP98M', 'SP98U']
close_prices = pd.DataFrame({
    'SP98H': sp98h_aligned['Close'],
    'SP98M': sp98m_aligned['Close'],
    'SP98U': sp98u_aligned['Close']
}).dropna()

open_prices = pd.DataFrame({
    'SP98H': sp98h_aligned['Open'],
    'SP98M': sp98m_aligned['Open'],
    'SP98U': sp98u_aligned['Open']
}).dropna()

# Align all to common dates
common_idx = close_prices.index
open_prices = open_prices.loc[common_idx]

n_bars = len(close_prices)
print(f"Common bars across all 3 contracts: {n_bars}")
print(f"Date range: {common_idx[0]} to {common_idx[-1]}")

# Point values — S&P 500 futures: $250 per point
point_values = pd.DataFrame(
    np.ones((n_bars, 3)) * 250,
    index=common_idx, columns=instruments
)

# Dividends — set to 0
dividends = pd.DataFrame(
    np.zeros((n_bars, 3)),
    index=common_idx, columns=instruments
)

# Equal allocation weights
alloc_weights = pd.DataFrame(
    np.ones((n_bars, 3)) / 3,
    index=common_idx, columns=instruments
)

# Transaction costs — 1 basis point
trans_costs = pd.Series(1e-4, index=instruments)

# Rebalance quarterly
rebalance_dates = list(common_idx[::63])

# --- Section 2.4.2: PCA Weights ---
print("\n--- PCA Weights ---")
returns = close_prices.pct_change().dropna()
cov = returns.cov().values
risk_dist = np.ones(3) / 3
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

# Load front-month contracts in order and stitch together
# Each contract is active for roughly one quarter
front_month_contracts = [
    ('SP98H', '1998-01-01', '1998-03-20'),
    ('SP98M', '1998-03-21', '1998-06-19'),
    ('SP98U', '1998-06-20', '1998-09-18'),
    ('SP98Z', '1998-09-19', '1998-12-18'),
    ('SP99H', '1998-12-19', '1999-03-19'),
    ('SP99M', '1999-03-20', '1999-06-18'),
    ('SP99U', '1999-06-19', '1999-09-17'),
    ('SP99Z', '1999-09-18', '1999-12-17'),
    ('SP00H', '1999-12-18', '2000-03-16'),
    ('SP00M', '2000-03-17', '2000-06-15'),
    ('SP00U', '2000-06-16', '2000-09-14'),
    ('SP00Z', '2000-09-15', '2000-12-14'),
]

pieces = []
for contract, start, end in front_month_contracts:
    df = load_contract(f'{contract}.txt')
    df = df[(df['Date'] >= start) & (df['Date'] <= end)].copy()
    df['Instrument'] = contract
    pieces.append(df)

futures_series = pd.concat(pieces).sort_values('Date').reset_index(drop=True)
futures_series = futures_series.set_index('Date')

print(f"Stitched futures series: {len(futures_series)} bars")
print(f"Date range: {futures_series.index[0]} to {futures_series.index[-1]}")
print(f"Contracts used: {futures_series['Instrument'].unique()}")

# Apply roll gap correction
dictio = {'Instrument': 'Instrument', 'Open': 'Open', 'Close': 'Close'}
rolled = mp.get_rolled_series(futures_series, dictio=dictio, match_end=True)
non_neg = mp.non_negative_rolled_prices(futures_series, dictio=dictio, match_end=True)

print("\nRolled series (first 10 rows):")
print(rolled.head(10))
print("\nNon-negative rolled prices (first 10 rows):")
print(non_neg[['Close', 'Returns', 'rPrices']].head(10))

# --- Figure 1: Price Series ---
fig1, axes1 = plt.subplots(3, 1, figsize=(14, 14))
fig1.suptitle("Chapter 2.4 — Multi-Product Series (Real S&P 500 Futures Data)",
              fontsize=14, y=0.98)

axes1[0].plot(result.index, result['K'], color='blue')
axes1[0].set_title("ETF Trick: $1 Investment Value Across 3 S&P 500 Futures Contracts (Section 2.4.1)")
axes1[0].set_xlabel("Date")
axes1[0].set_ylabel("Portfolio Value ($)")
axes1[0].tick_params(axis='x', rotation=45)

axes1[1].plot(futures_series.index, futures_series['Close'],
              color='red', label='Raw prices', linewidth=0.8)
axes1[1].plot(rolled.index, rolled['Close'],
              color='green', label='Rolled prices', linewidth=0.8)
axes1[1].set_title("Single Future Roll: Raw vs Rolled S&P 500 Futures Prices (Section 2.4.3)")
axes1[1].set_xlabel("Date")
axes1[1].set_ylabel("Price")
axes1[1].legend(loc='upper left')
axes1[1].tick_params(axis='x', rotation=45)

axes1[2].plot(non_neg.index, non_neg['rPrices'], color='purple', linewidth=0.8)
axes1[2].set_title("Single Future Roll: $1 Investment Value After Roll Correction (Section 2.4.3)")
axes1[2].set_xlabel("Date")
axes1[2].set_ylabel("Value ($)")
axes1[2].tick_params(axis='x', rotation=45)

plt.subplots_adjust(hspace=0.9, top=0.91, bottom=0.12)
plt.show()

# --- Figure 2: PCA Weights ---
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.suptitle("Chapter 2.4 — PCA Weights: Risk-Balanced Allocation\nAcross 3 S&P 500 Futures Contracts (Section 2.4.2)", 
              fontsize=12, y=0.98)
ax2.set_title("")  # no subplot title, everything in suptitle
weight_values = weights.flatten()
colors = ['green' if w > 0 else 'red' for w in weight_values]
bars = ax2.bar(instruments, weight_values, color=colors, width=0.5)
ax2.axhline(y=0, color='black', linewidth=0.8)
ax2.set_xlabel("Instrument")
ax2.set_ylabel("Weight")

for i, w in enumerate(weight_values):
    if abs(w) > 500:
        y_pos = w / 2   
        ax2.text(i, y_pos, f'{w:.2f}', ha='center', va='center',
                 fontsize=11, color='white', fontweight='bold')
    else:
        offset = 200 if w >= 0 else -400
        ax2.text(i, w + offset, f'{w:.2f}', ha='center', va='center',
                 fontsize=11, color='black', fontweight='bold')
        
plt.tight_layout()
plt.show()