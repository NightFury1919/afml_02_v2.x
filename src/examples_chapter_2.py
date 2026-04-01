import bars
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
n = 500
df = pd.DataFrame({
"Date": pd.date_range(start="2020-01-01", periods=n),
"Price": np.cumsum(np.random.randn(n)) + 100,
"Volume": np.random.randint(1, 50, size=n)
})



#df = df.iloc[:, 0:5]

df['Dollar'] = df['Price']*df['Volume']

# Price change & Labeling
df = bars.delta(df)
df = bars.labelling(df)


# CUSUM Filter
h = 2  # threshold - tune this to get more or fewer events
events = bars.cusum_filter(df, h)
print(f"CUSUM filter fired {len(events)} events")
print(events)

# exit() 
tick_imbalance_bar = bars.tick_imbalance_bars(df, expected_num_ticks_init=10)
print(tick_imbalance_bar)

time_bar = bars.time_bars(df)
print(time_bar)

# Initial conditions
p_b, p_s = bars.initial_conditions(df)
thresh_imbalance = 5
thresh_run = 3
# thresholds (tune these)
thresh_volume = 200
thresh_dollar = 10000

# generate bars
volume_bar = bars.volume_bars(df, thresh_volume)
dollar_bar = bars.dollar_bars(df, thresh_dollar)




print(volume_bar)
print(dollar_bar)

# Generate imbalance bars
imbalance_bar = bars.bar_gen(df, expected_num_ticks_init=50)
print(imbalance_bar)

run_bar = bars.bar_gen_run(df, thresh_run)
run_bar['Date'] = pd.to_datetime(run_bar['Date'])  
# Generate run bars
run_bar = bars.bar_gen_run(df, thresh_run)
print(run_bar)

thresh_tick = 25  # close a bar every 25 trades
tick_bar = bars.tick_bars(df, thresh_tick)
print(tick_bar)

# Plot bars
sns.set_style("whitegrid")

# Plot VWAP series
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle("VWAP Comparison by Bar Type")

axes[0,0].plot(time_bar['Date'], time_bar['Price'], color='brown')
axes[0,0].set_title("Time Bars")

axes[0,1].plot(tick_bar['Date'], tick_bar['Vwap'], color='purple')
axes[0,1].set_title("Tick Bars")

axes[1,0].plot(volume_bar['Date'], volume_bar['Vwap'], color='green')
axes[1,0].set_title("Volume Bars")

axes[1,1].plot(dollar_bar['Date'], dollar_bar['Vwap'], color='red')
axes[1,1].set_title("Dollar Bars")

axes[2,0].plot(tick_imbalance_bar['Date'], tick_imbalance_bar['Close'], color='blue')
axes[2,0].set_title("Tick Imbalance Bars")

axes[2,1].plot(run_bar['Date'], run_bar['Vwap'], color='orange')
axes[2,1].set_title("Run Bars")

plt.tight_layout()
plt.show()


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

#imbalance_bar = bars.bar_gen(df, thresh_imbalance)
#run_bar = bars.bar_gen_run(df, thresh_run)

#print(__file__)

file_path = os.path.dirname(__file__)
#print(file_path)

output_path = os.path.join(file_path, '..', 'output_data')
#print(output_path)


imbalance_bar.to_csv(os.path.join(output_path, 'imbalance_bars_output.csv'), index=False)
run_bar.to_csv(os.path.join(output_path, 'run_bars_output.csv'), index=False)
volume_bar.to_csv(os.path.join(output_path, 'volume_bars_output.csv'), index=False)
dollar_bar.to_csv(os.path.join(output_path, 'dollar_bars_output.csv'), index=False)


volume_bars = pd.read_csv(os.path.join(output_path, 'volume_bars_output.csv'))
dollar_bars = pd.read_csv(os.path.join(output_path, 'dollar_bars_output.csv'))

volume_bars['log_ret'] = np.log(volume_bars['Close']).diff().fillna(0)
dollar_bars['log_ret'] = np.log(dollar_bars['Close']).diff().fillna(0)


print("Saved imbalance bars to CSV")





