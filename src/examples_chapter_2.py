import bars
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import seaborn as sns
import scipy.stats as stats
#THIS IS WHAT I(ETHAN ROBINSON) HAVE ADDED. THIS IS WHERE YOU INSERT THE DATA FILE OR DATA TO TEST. 
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

# Initial conditions
p_b, p_s = bars.initial_conditions(df)
thresh_imbalance = 5
thresh_run = 50
# thresholds (tune these)
thresh_volume = 200
thresh_dollar = 10000

# generate bars
volume_bar = bars.volume_bars(df, thresh_volume)
dollar_bar = bars.dollar_bars(df, thresh_dollar)




print(volume_bar)
print(dollar_bar)

# Generate imbalance bars
imbalance_bar = bars.bar_gen(df, thresh_imbalance)
print(imbalance_bar)

# Generate run bars
run_bar = bars.bar_gen_run(df, thresh_run)
print(run_bar)

# Plot bars
sns.set_style("whitegrid")

plt.figure(figsize=(12,6))

# Plot VWAP series
plt.plot(imbalance_bar['Vwap'], label="Imbalance Bars VWAP")
plt.plot(run_bar['Vwap'], label="Run Bars VWAP")
plt.plot(volume_bar['Vwap'], label="Volume Bars VWAP")
plt.plot(dollar_bar['Vwap'], label="Dollar Bars VWAP")


# Labels (this is what you were missing)
plt.title("VWAP Comparison: Imbalance Bars vs Run Bars")
plt.xlabel("Bar Index (NOT Bar Index!)")
plt.ylabel("VWAP (Price)")
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

volume_bar_df = pd.read_csv(os.path.join(output_path, 'volume_bars_output.csv'))
dollar_bar_df = pd.read_csv(os.path.join(output_path, 'dollar_bars_output.csv'))


volume_bars = pd.read_csv(os.path.join(output_path, 'volume_bars_output.csv'))
dollar_bars = pd.read_csv(os.path.join(output_path, 'dollar_bars_output.csv'))

volume_bars['log_ret'] = np.log(volume_bars['Close']).diff().fillna(0)
dollar_bars['log_ret'] = np.log(dollar_bars['Close']).diff().fillna(0)





print("Saved imbalance bars to CSV")



