import numpy as np
import pandas as pd

def time_bars(df, freq='W'):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    bars = df.resample(freq).agg({
        'Price': 'last',
        'Volume': 'sum'
    }).dropna()

    return bars.reset_index()

def tick_bars(df, thresh):
    bars = []
    collector = []
    cumm_vol = 0
    vol_price = 0

    for i, (price, volume, date) in enumerate(zip(df['Price'], df['Volume'], df['Date'])):
        collector.append(price)
        cumm_vol += volume
        vol_price += price * volume

        if len(collector) >= thresh:
            open_p  = collector[0]
            high_p  = np.max(collector)
            low_p   = np.min(collector)
            close_p = collector[-1]
            vwap    = vol_price / cumm_vol

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))

            collector = []
            cumm_vol  = 0
            vol_price = 0

    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    return pd.DataFrame(bars, columns=cols)

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