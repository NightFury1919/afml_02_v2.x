import numpy as np
import pandas as pd
import os
import scipy.stats as stats

def ewma(arr, window):
    if len(arr) == 0:
        return 0

    alpha = 2 / (window + 1)
    ewma_val = arr[0]

    for i in range(1, len(arr)):
        ewma_val = alpha * arr[i] + (1 - alpha) * ewma_val

    return ewma_val

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
    pos_run = 0
    neg_run = 0

    cumm_vol = 0
    vol_price = 0
    collector = []
    bars = []

    for i, (label, price, date, volume) in enumerate(zip(df['Label'], df['Price'], df['Date'], df['Volume'])):

    # accumulate buys and sells independently - never reset mid-bar
        if label == 1:
            pos_run += 1
        elif label == -1:
            neg_run += 1

        theta = max(pos_run, neg_run)  # θ_T = max of both sides

        cumm_vol += volume
        vol_price += price * volume
        collector.append(price)

        if theta >= thresh:
            open_p = collector[0]
            high_p = np.max(collector)
            low_p = np.min(collector)
            close_p = collector[-1]
            vwap = vol_price / cumm_vol

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))

            # RESET everything including both run counters
            pos_run = 0
            neg_run = 0
            cumm_vol = 0
            vol_price = 0
            collector = []

    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    result = pd.DataFrame(bars, columns=cols)
    result['Date'] = pd.to_datetime(result['Date'])
    return result

def bar_gen(df, expected_num_ticks_init = 10, num_prev_bars=3):
    cum_theta = 0
    cumm_vol = 0
    vol_price = 0

    collector = []
    bars = []

    imbalance_array = []
    bar_lengths = []

    num_ticks = 0
    expected_num_ticks = expected_num_ticks_init
    expected_imbalance = 0

    for i, (label, price, date, volume) in enumerate(zip(df['Label'], df['Price'], df['Date'], df['Volume'])):

        # θ_T = Σ b_t * v_t
        imbalance = label * volume
        imbalance_array.append(imbalance)

        cum_theta += imbalance
        cumm_vol += volume
        vol_price += price * volume
        collector.append(price)

        num_ticks += 1

        # Initialize expected imbalance
        if len(bars) == 0 and len(imbalance_array) >= expected_num_ticks_init:
            expected_imbalance = max(
                ewma(imbalance_array, expected_num_ticks_init),
                1e-6
            )

        # AFML stopping condition
        if expected_imbalance != 0 and abs(cum_theta) >= expected_num_ticks * abs(expected_imbalance):

            open_p = collector[0]
            high_p = np.max(collector)
            low_p = np.min(collector)
            close_p = collector[-1]
            vwap = vol_price / cumm_vol

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))

            bar_lengths.append(num_ticks)

            # RESET
            cum_theta = 0
            cumm_vol = 0
            vol_price = 0
            collector = []
            num_ticks = 0

            # Update expectations (EWMA formulas from book)
            expected_num_ticks = ewma(bar_lengths, num_prev_bars)
            expected_imbalance = max(
    ewma(
        imbalance_array,
        max(1, int(num_prev_bars * expected_num_ticks))
    ),
    1e-6
)

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


def tick_imbalance_bars(df, expected_num_ticks_init = 10, num_prev_bars=3):
    cum_theta = 0
    collector = []
    bars = []

    imbalance_array = []
    bar_lengths = []

    num_ticks = 0
    expected_num_ticks = expected_num_ticks_init
    expected_imbalance = 0

    for i, (label, price, date) in enumerate(zip(df['Label'], df['Price'], df['Date'])):

        imbalance = label  # v_t = 1
        imbalance_array.append(imbalance)

        cum_theta += imbalance
        collector.append(price)
        num_ticks += 1

        if len(bars) == 0 and len(imbalance_array) >= expected_num_ticks_init:
            expected_imbalance = max(
                ewma(imbalance_array, expected_num_ticks_init),
                1e-6
            )
    

        if expected_imbalance != 0 and abs(cum_theta) >= expected_num_ticks * abs(expected_imbalance):

            open_p = collector[0]
            high_p = np.max(collector)
            low_p = np.min(collector)
            close_p = collector[-1]

            bars.append((date, i, open_p, low_p, high_p, close_p))

            bar_lengths.append(num_ticks)

            cum_theta = 0
            collector = []
            num_ticks = 0

            expected_num_ticks = ewma(bar_lengths, num_prev_bars)
            expected_imbalance = max(
    ewma(
        imbalance_array,
        max(1, int(num_prev_bars * expected_num_ticks))
    ),
    1e-6
)

    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close']
    return pd.DataFrame(bars, columns=cols)

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

        s_pos = max(0, s_pos + delta)   # grow upward or reset to zero
        s_neg = min(0, s_neg + delta)   # grow downward or reset to zero

        if s_pos >= h:                  # upward drift exceeded threshold
            s_pos = 0                   # reset
            events.append(date)         # record this date as an event

        elif abs(s_neg) >= h:           # downward drift exceeded threshold
            s_neg = 0                   # reset
            events.append(date)         # record this date as an event

    return pd.DatetimeIndex(events)

