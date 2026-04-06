import numpy as np
import pandas as pd

def cusum_filter(df, h):
    events = []         # list of dates where a signal fired
    s_pos = 0           # S+ accumulator (tracks upward drift)
    s_neg = 0           # S- accumulator (tracks downward drift)

    # compute daily price changes
    prices = df['Price']
    diff = prices.diff()  # difference between each price and the previous one

    for date, delta in zip(df['Date'], diff):
        if pd.isna(delta):      # skip the first row, which has no previous price
            continue
        # Symmetric CUSUM Filter
        s_pos = max(0, s_pos + delta)   # grow upward or reset to zero
        s_neg = min(0, s_neg + delta)   # grow downward or reset to zero

        if s_pos >= h:                  # upward drift exceeded threshold
            s_pos = 0                   # reset
            events.append(date)         # record this date as an event

        elif abs(s_neg) >= h:           # downward drift exceeded threshold
            s_neg = 0                   # reset
            events.append(date)         # record this date as an event

    return pd.DatetimeIndex(events)
