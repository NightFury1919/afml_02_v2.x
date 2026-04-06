import bars
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# --- Data Generation ---
np.random.seed(42)
n = 500
df = pd.DataFrame({
    "Date": pd.date_range(start="2020-01-01", periods=n),
    "Price": np.cumsum(np.random.randn(n)) + 100,
    "Volume": np.random.randint(1, 50, size=n)
})

df['Dollar'] = df['Price'] * df['Volume']
df = bars.delta(df)
df = bars.tick_rule(df)

# --- Initial Conditions ---
p_b, p_s = bars.estimate_buy_sell_probs(df)

# --- CUSUM Filter ---
h = 2
events = bars.cusum_filter(df, h)
print(f"CUSUM filter fired {len(events)} events")
print(events)

# --- Standard Bars ---
time_bar        = bars.time_bars(df)
tick_bar        = bars.tick_bars(df, thresh=25)
volume_bar      = bars.volume_bars(df, thresh=200)
dollar_bar      = bars.dollar_bars(df, thresh=10000)

# --- Information-Driven Bars ---
tick_imbalance_bar  = bars.tick_imbalance_bars(df, expected_num_ticks_init=5)
imbalance_bar       = bars.volume_imbalance_bars(df, expected_num_ticks_init=10)
run_bar             = bars.tick_run_bars(df, expected_num_ticks_init=10)
volume_run_bar      = bars.volume_run_bars(df, expected_num_ticks_init=10)

# --- Print all bars ---
print(time_bar)
print(tick_bar)
print(volume_bar)
print(dollar_bar)
print(tick_imbalance_bar)
print(imbalance_bar)
print(run_bar)
print(volume_run_bar)

# --- Plot: Bar Type Comparison ---
sns.set_style("whitegrid")
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
fig.suptitle("VWAP Comparison by Bar Type", fontsize=16, y=0.98)

plots = [
    (axes[0,0], time_bar['Date'],           time_bar['Price'],           'brown',   "Time Bars"),
    (axes[0,1], tick_bar['Date'],           tick_bar['Vwap'],            'purple',  "Tick Bars"),
    (axes[1,0], volume_bar['Date'],         volume_bar['Vwap'],          'green',   "Volume Bars"),
    (axes[1,1], dollar_bar['Date'],         dollar_bar['Vwap'],          'red',     "Dollar Bars"),
    (axes[2,0], tick_imbalance_bar['Date'], tick_imbalance_bar['Close'], 'blue',    "Tick Imbalance Bars"),
    (axes[2,1], imbalance_bar['Date'],      imbalance_bar['Vwap'],       'teal',    "Volume Imbalance Bars"),
    (axes[3,0], run_bar['Date'],            run_bar['Vwap'],             'orange',  "Tick Run Bars"),
    (axes[3,1], volume_run_bar['Date'],     volume_run_bar['Vwap'],      'magenta', "Volume Run Bars"),
]

for ax, x, y, color, title in plots:
    ax.plot(x, y, color=color)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Price", fontsize=9)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlim(pd.Timestamp('2020-01-01'), pd.Timestamp('2021-05-14'))

plt.subplots_adjust(hspace=0.7, wspace=0.3)
plt.show()

# --- Plot: CUSUM Events ---
plt.figure(figsize=(12, 4))
plt.plot(df['Date'], df['Price'], color='gray', label='Price')
plt.scatter(events.to_series(),
            df.set_index('Date').loc[events, 'Price'],
            color='red', zorder=5, label='CUSUM Events')
plt.title("CUSUM Filter Events on Price Series")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# --- Save CSVs ---
file_path   = os.path.dirname(__file__)
output_path = os.path.join(file_path, '..', 'output_data')

imbalance_bar.to_csv(os.path.join(output_path, 'imbalance_bars_output.csv'),    index=False)
run_bar.to_csv(os.path.join(output_path, 'run_bars_output.csv'),                index=False)
volume_bar.to_csv(os.path.join(output_path, 'volume_bars_output.csv'),          index=False)
dollar_bar.to_csv(os.path.join(output_path, 'dollar_bars_output.csv'),          index=False)

volume_bars_df = pd.read_csv(os.path.join(output_path, 'volume_bars_output.csv'))
dollar_bars_df = pd.read_csv(os.path.join(output_path, 'dollar_bars_output.csv'))

volume_bars_df['log_ret'] = np.log(volume_bars_df['Close']).diff().fillna(0)
dollar_bars_df['log_ret'] = np.log(dollar_bars_df['Close']).diff().fillna(0)

print("Saved bars to CSV")