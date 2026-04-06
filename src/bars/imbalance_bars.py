import numpy as np
import pandas as pd
from .utils import ewma

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
        # Tick Imbalance Accumulation:
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
    
        #Tick Imbalance Bar stopping formula: this if statement is the formula
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

def volume_imbalance_bars(df, expected_num_ticks_init = 10, num_prev_bars=3):
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

        # Volume/Dollar imbalance accumulation: the imbalance = and cum_theta += imbalance
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

        # V/D Imbalance Bar stopping rule if statement
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

            # V/D Expected Imbalance Equation, from here until 1e-6
            expected_num_ticks = ewma(bar_lengths, num_prev_bars)
            expected_imbalance = max(
    ewma(
        imbalance_array,
        max(1, int(num_prev_bars * expected_num_ticks))
    ),
    1e-6
)

    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    result = pd.DataFrame(bars, columns=cols)
    result['Date'] = pd.to_datetime(result['Date'])
    return result