import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bars

# --- Load Real Binance Tick Data ---
base = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base, 'input_data', 'BTCTUSD-trades-2026-03.csv')

print("Loading data...")
raw = pd.read_csv(data_path, header=None,
                  names=['TradeID', 'Price', 'Volume', 'QuoteVolume',
                         'Timestamp', 'IsBuyerMaker', 'IsBestMatch'])

# Convert timestamp from microseconds to datetime
raw['Date'] = pd.to_datetime(raw['Timestamp'], unit='us')

# Convert IsBuyerMaker to tick direction Label
# IsBuyerMaker=True means seller was aggressive = price went down = -1
# IsBuyerMaker=False means buyer was aggressive = price went up = +1
raw['Label'] = raw['IsBuyerMaker'].apply(lambda x: -1 if x else 1)

# Build the dataframe in the format your bar functions expect
df = raw[['Date', 'Price', 'Volume', 'Label']].copy()
df['Dollar'] = df['Price'] * df['Volume']

# Compute Delta (price changes between consecutive trades)
df = bars.delta(df)

print(f"Loaded {len(df)} trades")
print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
print()

# --- Standard Bars ---
print("Generating standard bars...")
time_bar   = bars.time_bars(df, freq='D')        # daily bars
tick_bar   = bars.tick_bars(df, thresh=100)       # every 100 trades
volume_bar = bars.volume_bars(df, thresh=1.0)     # every 1 BTC traded
dollar_bar = bars.dollar_bars(df, thresh=50000)   # every $50,000 traded

print(f"Time bars:   {len(time_bar)} bars")
print(f"Tick bars:   {len(tick_bar)} bars")
print(f"Volume bars: {len(volume_bar)} bars")
print(f"Dollar bars: {len(dollar_bar)} bars")

# --- Information-Driven Bars ---
print("\nGenerating information-driven bars...")
tick_imbalance_bar = bars.tick_imbalance_bars(df, expected_num_ticks_init=500)
imbalance_bar      = bars.volume_imbalance_bars(df, expected_num_ticks_init=50)
run_bar            = bars.tick_run_bars(df, expected_num_ticks_init=50)
volume_run_bar = bars.volume_run_bars(df, expected_num_ticks_init=10)
print(f"Volume run bar expected imbalance debug: {len(volume_run_bar)} bars")

print(f"Tick imbalance bars:  {len(tick_imbalance_bar)} bars")
print(f"Volume imbalance bars:{len(imbalance_bar)} bars")
print(f"Tick run bars:        {len(run_bar)} bars")
print(f"Volume run bars:      {len(volume_run_bar)} bars")

# --- CUSUM Filter ---
print("\nApplying CUSUM filter...")
h = 500  # threshold in dollars
events = bars.cusum_filter(df, h)
print(f"CUSUM filter fired {len(events)} events")

# --- Plot 1: Standard Bars ---
sns.set_style("whitegrid")
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
fig1.suptitle("Chapter 2 — Standard Bar Types — Real Bitcoin Trade Data (March 2026)",
              fontsize=13, y=0.98)

standard_plots = [
    (axes1[0,0], time_bar['Date'],   time_bar['Price'],   'brown',  "Time Bars (Daily)"),
    (axes1[0,1], tick_bar['Date'],   tick_bar['Vwap'],    'purple', "Tick Bars (every 100 trades)"),
    (axes1[1,0], volume_bar['Date'], volume_bar['Vwap'],  'green',  "Volume Bars (every 1 BTC)"),
    (axes1[1,1], dollar_bar['Date'], dollar_bar['Vwap'],  'red',    "Dollar Bars (every $50,000)"),
]

for ax, x, y, color, title in standard_plots:
    ax.plot(x, y, color=color)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Price (USD)", fontsize=9)
    ax.tick_params(axis='x', labelsize=8, rotation=45)

plt.subplots_adjust(hspace=0.6, wspace=0.3)
plt.show()

# --- Plot 2: Information-Driven Bars ---
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle("Chapter 2 — Information-Driven Bar Types — Real Bitcoin Trade Data (March 2026)",
              fontsize=13, y=0.98)

info_plots = [
    (axes2[0,0], tick_imbalance_bar['Date'], tick_imbalance_bar['Close'], 'blue',    "Tick Imbalance Bars"),
    (axes2[0,1], imbalance_bar['Date'],      imbalance_bar['Vwap'],       'teal',    "Volume Imbalance Bars"),
    (axes2[1,0], run_bar['Date'],            run_bar['Vwap'],             'orange',  "Tick Run Bars"),
    (axes2[1,1], volume_run_bar['Date'],     volume_run_bar['Vwap'],      'magenta', "Volume Run Bars"),
]

for ax, x, y, color, title in info_plots:
    ax.plot(x, y, color=color)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Price (USD)", fontsize=9)
    ax.tick_params(axis='x', labelsize=8, rotation=45)

plt.subplots_adjust(hspace=0.6, wspace=0.3)
plt.show()