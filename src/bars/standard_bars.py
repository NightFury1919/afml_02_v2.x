import numpy as np
import pandas as pd

# Standard Bars — AFML Chapter 2, Section 2.3.1
# All four standard bar types sample at fixed thresholds rather than
# adapting to market activity. They serve as a baseline comparison
# for the information-driven bars in Section 2.3.2.
# VWAP (Volume Weighted Average Price) is computed in all bar types:
#   VWAP = Σ(price * volume) / Σ(volume)

def time_bars(df, freq='W'):
    # Time Bars — sample at fixed time intervals.
    # Each bar represents one period (e.g. weekly) regardless of trading activity.
    # The book notes time bars oversample quiet periods and undersample active ones.
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    bars = df.resample(freq).agg({
        'Price': 'last',
        'Volume': 'sum'
    }).dropna()

    return bars.reset_index()

def tick_bars(df, thresh):
    # Tick Bars — sample every N trades regardless of price or volume.
    # Addresses the limitation of time bars by sampling based on trading activity.
    # A bar closes when len(collector) >= thresh (i.e. N trades have occurred).
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
            # VWAP = Σ(price * volume) / Σ(volume)
            vwap    = vol_price / cumm_vol

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))

            collector = []
            cumm_vol  = 0
            vol_price = 0

    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    return pd.DataFrame(bars, columns=cols)

def volume_bars(df, thresh):
    # Volume Bars — sample every time a fixed amount of the asset is traded.
    # Improves on tick bars by accounting for trade size differences.
    # A bar closes when cumm_vol >= thresh (i.e. N units have been traded).
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
            # VWAP = Σ(price * volume) / Σ(volume)
            vwap = vol_price / cumm_vol

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))

            # reset
            cumm_vol = 0
            vol_price = 0
            collector = []

    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    return pd.DataFrame(bars, columns=cols)

def dollar_bars(df, thresh):
    # Dollar Bars — sample every time a fixed dollar value is traded.
    # Most robust to price-level changes and corporate actions.
    # A bar closes when cumm_dollar >= thresh (i.e. $N has been exchanged).
    # The book notes dollar bars are the most useful standard bar type
    # and uses them throughout subsequent chapters.
    cumm_dollar = 0
    cumm_vol = 0
    vol_price = 0
    collector = []
    bars = []

    for i, (price, volume, date) in enumerate(zip(df['Price'], df['Volume'], df['Date'])):
        # dollar = price * volume for this trade
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
            # VWAP = Σ(price * volume) / Σ(volume)
            vwap = vol_price / cumm_vol

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))

            # reset
            cumm_dollar = 0
            cumm_vol = 0
            vol_price = 0
            collector = []

    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    return pd.DataFrame(bars, columns=cols)
