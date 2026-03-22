import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import seaborn as sns
import scipy.stats as stats


def add(a, b):
    result = a + b
    print(f"result: {result}, {a} {b}")
    return result



def delta(df):
    a = np.diff(df['Price'])
    a = np.insert(a, 0, 0)
    df['Delta'] = a
    return df

def labelling(df):
    b = np.ones(len(df['Price']))
    for i, delta in enumerate(df['Delta']):
        if i > 0:
            if delta == 0:
                b[i] = b[i-1]
            else:
                b[i] = abs(delta) / delta
    df['Label'] = b
    return df

def initial_conditions(df):
    prob = pd.DataFrame(pd.pivot_table(df, index='Label', values='Price', aggfunc='count'))
    prob = np.array(prob)
    p_b = prob[1]/(prob[0]+prob[1])
    p_s = prob[0]/(prob[0]+prob[1])
    return p_b, p_s

def bar_gen_run(df, thresh):
    cumm, open, low, high, close, cumm_vol, vol_price, b, s = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    collector, bar, thresh_buffer = [], [], []

    for i, (label, price, date, volume) in enumerate(zip(df['Label'], df['Price'], df['Date'], df['Volume'])):
        if label == 1:
            b = b + label
        else:
            s = s + label
        theta = max(b, abs(s))

        cumm_vol = cumm_vol + volume
        vol_price = vol_price + (price * volume)
        collector.append(price)
        if theta >= thresh:
            open = collector[0]
            high = np.max(collector)
            low = np.min(collector)
            close = collector[-1]
            vwap = vol_price / cumm_vol
            bar.append((date, i, open, low, high, close, vwap))
            a = len(collector) * max(((b/len(collector)), (1-(b/len(collector)))))
            thresh_buffer.append(a)
            if i > 500000: thresh = np.average(thresh_buffer)
            theta, open, low, high, close, cumm_vol, vol_price, b, s = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            collector = []
    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    run_bar = pd.DataFrame(bar, columns= cols)
    return run_bar

def bar_gen(df, thresh):
    cumm, open, low, high, close, cumm_vol, vol_price, b = 0, 0, 0, 0, 0, 0, 0, 0
    collector, bar, thresh_buffer = [], [], []

    for i, (label, price, date, volume) in enumerate(zip(df['Label'], df['Price'], df['Date'], df['Volume'])):
        if label == 1:
            b = b + 1
        cumm = cumm + label
        cumm_vol = cumm_vol + volume
        vol_price = vol_price + (price * volume)
        collector.append(price)
        if abs(cumm) >= thresh:
            open = collector[0]
            high = np.max(collector)
            low = np.min(collector)
            close = collector[-1]
            vwap = vol_price / cumm_vol
            bar.append((date, i, open, low, high, close, vwap))
            a = len(collector) * abs((2*(b/len(collector)))-1)
            thresh_buffer.append(a)
            if i > 500000: thresh = np.average(thresh_buffer)
            cumm, open, low, high, close, cumm_vol, vol_price, b = 0, 0, 0, 0, 0, 0, 0, 0
            collector = []
    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    imbalance_bar = pd.DataFrame(bar, columns= cols)
    return imbalance_bar

def volume_bars(df, thresh):
    cumm_vol = 0
    vol_price = 0
    collector = []
    bars = []

    for i, (price, volume, date) in enumerate(zip(df['Price'], df['Volume'], df['Date'])):
        cumm_vol += volume
        vol_price += price * volume
        collector.append(price)

        if cumm_vol >= thresh:
            open_p = collector[0]
            high_p = np.max(collector)
            low_p = np.min(collector)
            close_p = collector[-1]
            vwap = vol_price / cumm_vol

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))

            # reset
            cumm_vol = 0
            vol_price = 0
            collector = []

    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    return pd.DataFrame(bars, columns=cols)

def dollar_bars(df, thresh):
    cumm_dollar = 0
    cumm_vol = 0
    vol_price = 0
    collector = []
    bars = []

    for i, (price, volume, date) in enumerate(zip(df['Price'], df['Volume'], df['Date'])):
        dollar = price * volume
        cumm_dollar += dollar
        cumm_vol += volume
        vol_price += dollar
        collector.append(price)

        if cumm_dollar >= thresh:
            open_p = collector[0]
            high_p = np.max(collector)
            low_p = np.min(collector)
            close_p = collector[-1]
            vwap = vol_price / cumm_vol

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))

            # reset
            cumm_dollar = 0
            cumm_vol = 0
            vol_price = 0
            collector = []

    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    return pd.DataFrame(bars, columns=cols)


'''
if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import os
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
    df = delta(df)
    df = labelling(df)

    # Initial conditions
    p_b, p_s = initial_conditions(df)
    thresh_imbalance = 5
    thresh_run = 50
    # thresholds (tune these)
    thresh_volume = 200
    thresh_dollar = 10000

    # generate bars
    volume_bar = volume_bars(df, thresh_volume)
    dollar_bar = dollar_bars(df, thresh_dollar)
    



    print(volume_bar)
    print(dollar_bar)

    # Generate imbalance bars
    imbalance_bar = bar_gen(df, thresh_imbalance)
    print(imbalance_bar)

    # Generate run bars
    run_bar = bar_gen_run(df, thresh_run)
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
    plt.xlabel("Bar Index (NOT Time!)")
    plt.ylabel("VWAP (Price)")
    plt.legend()

    plt.show()

    #imbalance_bar = bar_gen(df, thresh_imbalance)
    #run_bar = bar_gen_run(df, thresh_run)

    output_path = r'C:\ws\chapter_two_working_code\output_data'

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

'''

