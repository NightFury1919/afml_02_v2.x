import numpy as np
import pandas as pd

def cusum_filter(df, h):
    # Symmetric CUSUM Filter — AFML Chapter 2, Section 2.5.2, page 39
    # (Snippet 2.4 in the book)
    #
    # A quality-control method that detects when a price series has drifted
    # significantly in either direction from a reset level of zero.
    # Fires an event when cumulative drift exceeds threshold h, then resets.
    # Avoids triggering multiple events when price hovers near a threshold
    # (unlike Bollinger Bands).
    #
    # Formulas (page 39):
    #   S+_t = max{0, S+_{t-1} + y_t - E_{t-1}[y_t]},  S+_0 = 0
    #   S-_t = min{0, S-_{t-1} + y_t - E_{t-1}[y_t]},  S-_0 = 0
    #   S_t  = max{S+_t, -S-_t}
    #
    # An event fires at t if S_t >= h, at which point S+_t and S-_t are reset to 0.
    # Here E_{t-1}[y_t] = y_{t-1} (previous price), so y_t - E_{t-1}[y_t] = Δp_t

    events = []   # list of dates where a signal fired
    s_pos = 0     # S+_t accumulator (tracks upward drift)
    s_neg = 0     # S-_t accumulator (tracks downward drift)

    # Compute Δp_t — price change between consecutive bars
    prices = df['Price']
    diff = prices.diff()

    for date, delta in zip(df['Date'], diff):
        if pd.isna(delta):  # skip the first row (no previous price)
            continue

        # S+_t = max{0, S+_{t-1} + Δp_t} — grows on upticks, resets to 0 on downticks
        s_pos = max(0, s_pos + delta)
        # S-_t = min{0, S-_{t-1} + Δp_t} — grows on downticks, resets to 0 on upticks
        s_neg = min(0, s_neg + delta)

        if s_pos >= h:
            # Upward drift exceeded threshold h — fire event and reset S+
            s_pos = 0
            events.append(date)

        elif abs(s_neg) >= h:
            # Downward drift exceeded threshold h — fire event and reset S-
            s_neg = 0
            events.append(date)

    return pd.DatetimeIndex(events)
