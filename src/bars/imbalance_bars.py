import numpy as np
import pandas as pd
from .utils import ewma

# Information-Driven Imbalance Bars — AFML Chapter 2, Section 2.3.2
# Imbalance bars close when the accumulated signed imbalance exceeds
# an adaptive expected threshold, causing more frequent sampling during
# periods of informed trading and less frequent sampling during quiet periods.

def tick_imbalance_bars(df, expected_num_ticks_init=10, num_prev_bars=3):
    # Tick Imbalance Bars (TIBs) — AFML Chapter 2, page 29
    # Uses tick direction only (v_t = 1 for every trade).
    cum_theta = 0
    collector = []
    bars = []

    imbalance_array = []
    bar_lengths = []

    num_ticks = 0
    expected_num_ticks = expected_num_ticks_init
    expected_imbalance = 0

    for i, (label, price, date) in enumerate(zip(df['Label'], df['Price'], df['Date'])):

        # Tick Imbalance Accumulation — page 29
        # θ_T = Σ b_t, from t = 1 to T
        # Each tick contributes +1 (buy) or -1 (sell) to the running total
        imbalance = label  # b_t, v_t = 1 for tick imbalance
        imbalance_array.append(imbalance)
        cum_theta += imbalance  # θ_T = running sum of b_t

        collector.append(price)
        num_ticks += 1

        # Initialize expected imbalance using EWMA over warmup period
        # E0[θ_T] = E0[T] * |2P[b_t=1] - 1|  (page 29)
        if len(bars) == 0 and len(imbalance_array) >= expected_num_ticks_init:
            expected_imbalance = max(
                ewma(imbalance_array, expected_num_ticks_init),
                1e-6
            )

        # Tick Imbalance Bar Stopping Formula — page 29
        # T* = arg min_T { |θ_T| >= E0[T] * |2P[b_t=1] - 1| }
        # Close the bar when actual imbalance exceeds the expected imbalance
        if expected_imbalance != 0 and abs(cum_theta) >= expected_num_ticks * abs(expected_imbalance):

            open_p = collector[0]
            high_p = np.max(collector)
            low_p = np.min(collector)
            close_p = collector[-1]

            bars.append((date, i, open_p, low_p, high_p, close_p))
            bar_lengths.append(num_ticks)

            # Reset accumulators
            cum_theta = 0
            collector = []
            num_ticks = 0

            # Tick Imbalance Bar Expected Imbalance update — page 29
            # Update E0[T] and E0[|2P[b_t=1] - 1|] using EWMA of prior bars
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


def volume_imbalance_bars(df, expected_num_ticks_init=10, num_prev_bars=3):
    # Volume/Dollar Imbalance Bars (VIBs/DIBs) — AFML Chapter 2, page 30
    # Extends tick imbalance bars by weighting each tick by its volume (v_t).
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

        # Volume/Dollar Imbalance Accumulation — page 30
        # θ_T = Σ b_t * v_t, from t = 1 to T
        # Each tick contributes its signed volume to the running total
        imbalance = label * volume  # b_t * v_t
        imbalance_array.append(imbalance)
        cum_theta += imbalance  # θ_T = running sum of b_t * v_t

        cumm_vol += volume
        vol_price += price * volume
        collector.append(price)
        num_ticks += 1

        # Initialize expected imbalance using EWMA over warmup period
        # E0[θ_T] = E0[T] * (2v+ - E0[v_t])  (page 30)
        if len(bars) == 0 and len(imbalance_array) >= expected_num_ticks_init:
            expected_imbalance = max(
                ewma(imbalance_array, expected_num_ticks_init),
                1e-6
            )

        # V/D Imbalance Bar Stopping Rule — page 30
        # T* = arg min_T { |θ_T| >= E0[T] * |2v+ - E0[v_t]| }
        # Close the bar when actual volume imbalance exceeds expected
        if expected_imbalance != 0 and abs(cum_theta) >= expected_num_ticks * abs(expected_imbalance):

            open_p = collector[0]
            high_p = np.max(collector)
            low_p = np.min(collector)
            close_p = collector[-1]
            # VWAP = Σ(price * volume) / Σ(volume)
            vwap = vol_price / cumm_vol

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))
            bar_lengths.append(num_ticks)

            # Reset accumulators
            cum_theta = 0
            cumm_vol = 0
            vol_price = 0
            collector = []
            num_ticks = 0

            # V/D Expected Imbalance Equation update — page 30
            # E0[θ_T] = E0[T] * (2v+ - E0[v_t])
            # Update E0[T] and (2v+ - E0[v_t]) using EWMA of prior bars
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
