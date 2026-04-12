import numpy as np
import pandas as pd
from .utils import ewma

def volume_run_bars(df, expected_num_ticks_init=10, num_prev_bars=3):
    pos_run = 0   # accumulated buy volume
    neg_run = 0   # accumulated sell volume
    cumm_vol = 0
    vol_price = 0
    collector = []
    bars = []

    bar_lengths = []
    buy_vol_proportions = []  # E0[v_t | b_t=1] tracker
    sell_vol_proportions = [] # E0[v_t | b_t=-1] tracker
    num_ticks = 0

    expected_num_ticks = expected_num_ticks_init
    expected_p_buy = 0.5
    expected_buy_vol = 0.01    # more realistic for BTC
    expected_sell_vol = 0.01   # more realistic for BTC
    expected_imbalance = 0

    for i, (label, price, date, volume) in enumerate(zip(df['Label'], df['Price'], df['Date'], df['Volume'])):

        # Volume/Dollar Run Bar length definition
        if label == 1:
            pos_run += volume   # Σ b_t*v_t where b_t=1
        elif label == -1:
            neg_run += volume   # -Σ b_t*v_t where b_t=-1

        # θ_T = max of both sides
        theta = max(pos_run, neg_run)

        cumm_vol += volume
        vol_price += price * volume
        collector.append(price)
        num_ticks += 1

        # initialize expected imbalance after warmup
        if len(bars) == 0 and num_ticks >= expected_num_ticks_init:
            expected_imbalance = max(
                expected_num_ticks * max(
                    expected_p_buy * expected_buy_vol,
                    (1 - expected_p_buy) * expected_sell_vol
                ),
                1e-6
            )

        
        # Volume/Dollar Run Bar stopping rule
        if expected_imbalance != 0 and theta >= expected_imbalance:
            open_p  = collector[0]
            high_p  = np.max(collector)
            low_p   = np.min(collector)
            close_p = collector[-1]
            vwap    = vol_price / cumm_vol

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))
            bar_lengths.append(num_ticks)

            # track buy/sell volume averages for EWMA updates
            buy_vol_proportions.append(pos_run / num_ticks if num_ticks > 0 else expected_buy_vol)
            sell_vol_proportions.append(neg_run / num_ticks if num_ticks > 0 else expected_sell_vol)

            # reset
            pos_run = 0
            neg_run = 0
            cumm_vol = 0
            vol_price = 0
            collector = []
            num_ticks = 0

            # Volume/Dollar Run Bar expected imbalance 
            expected_num_ticks = ewma(bar_lengths, num_prev_bars)
            expected_p_buy = ewma([b / (b + s) if (b + s) > 0 else 0.5
                                   for b, s in zip(buy_vol_proportions, sell_vol_proportions)],
                                  num_prev_bars)
            expected_buy_vol = ewma(buy_vol_proportions, num_prev_bars)
            expected_sell_vol = ewma(sell_vol_proportions, num_prev_bars)
            expected_imbalance = max(
                expected_num_ticks * max(
                    expected_p_buy * expected_buy_vol,
                    (1 - expected_p_buy) * expected_sell_vol
                ),
                1e-6
            )

    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    result = pd.DataFrame(bars, columns=cols)
    result['Date'] = pd.to_datetime(result['Date'])
    return result

def tick_run_bars(df, expected_num_ticks_init=10, num_prev_bars=3):
    pos_run = 0
    neg_run = 0
    cumm_vol = 0
    vol_price = 0
    collector = []
    bars = []

    bar_lengths = []
    buy_tick_proportions = []  # stores proportion of buy ticks per bar
    num_ticks = 0

    expected_num_ticks = expected_num_ticks_init
    expected_p_buy = 0.5       # initial guess: 50% buys, 50% sells
    expected_imbalance = 0

    for i, (label, price, date, volume) in enumerate(zip(df['Label'], df['Price'], df['Date'], df['Volume'])):
        # Tick run bar definition
        if label == 1:
            pos_run += 1
        elif label == -1:
            neg_run += 1

        theta = max(pos_run, neg_run)

        cumm_vol += volume
        vol_price += price * volume
        collector.append(price)
        num_ticks += 1

        # initialize expected imbalance after warmup period
        if len(bars) == 0 and num_ticks >= expected_num_ticks_init:
            expected_imbalance = max(
                expected_num_ticks * max(expected_p_buy, 1 - expected_p_buy),
                1e-6
            )

        
        # Tick run bar stopping definition
        if expected_imbalance != 0 and theta >= expected_imbalance:
            open_p  = collector[0]
            high_p  = np.max(collector)
            low_p   = np.min(collector)
            close_p = collector[-1]
            vwap    = vol_price / cumm_vol

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))
            bar_lengths.append(num_ticks)

            # proportion of buy ticks in this bar
            buy_proportion = pos_run / num_ticks if num_ticks > 0 else 0.5
            buy_tick_proportions.append(buy_proportion)

            # reset
            pos_run = 0
            neg_run = 0
            cumm_vol = 0
            vol_price = 0
            collector = []
            num_ticks = 0

            # Tick Run Bar expected imbalance
            expected_num_ticks = ewma(bar_lengths, num_prev_bars)
            expected_p_buy = ewma(buy_tick_proportions, num_prev_bars)
            expected_imbalance = max(
                expected_num_ticks * max(expected_p_buy, 1 - expected_p_buy),
                1e-6
            )

    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    result = pd.DataFrame(bars, columns=cols)
    result['Date'] = pd.to_datetime(result['Date'])
    return result