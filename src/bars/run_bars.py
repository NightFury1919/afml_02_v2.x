import numpy as np
import pandas as pd
from .utils import ewma

# Information-Driven Run Bars — AFML Chapter 2, Section 2.3.2
# Run bars monitor sequences of trades in the same direction (runs),
# sampling more frequently when large traders are sweeping the order book.
# Unlike imbalance bars which track net imbalance, run bars track the
# maximum of cumulative buys vs cumulative sells independently.

def tick_run_bars(df, expected_num_ticks_init=10, num_prev_bars=3):
    # Tick Run Bars (TRBs) — AFML Chapter 2, page 31
    # Counts raw ticks on each side (buy ticks vs sell ticks).
    pos_run = 0  # cumulative count of buy ticks in current bar
    neg_run = 0  # cumulative count of sell ticks in current bar
    cumm_vol = 0
    vol_price = 0
    collector = []
    bars = []

    bar_lengths = []
    buy_tick_proportions = []  # tracks P[b_t=1] per bar for EWMA update
    num_ticks = 0

    expected_num_ticks = expected_num_ticks_init
    expected_p_buy = 0.5       # initial guess: 50% buys, 50% sells
    expected_imbalance = 0

    for i, (label, price, date, volume) in enumerate(zip(df['Label'], df['Price'], df['Date'], df['Volume'])):

        # Tick Run Bar Length Definition — page 31
        # θ_T = max( Σ b_t where b_t=1, -Σ b_t where b_t=-1 )
        # Accumulate buy and sell ticks independently — never reset mid-bar
        if label == 1:
            pos_run += 1   # count buy tick
        elif label == -1:
            neg_run += 1   # count sell tick

        # θ_T = max of both sides
        theta = max(pos_run, neg_run)

        cumm_vol += volume
        vol_price += price * volume
        collector.append(price)
        num_ticks += 1

        # Initialize expected imbalance using warmup period
        # E0[θ_T] = E0[T] * max{P[b_t=1], 1 - P[b_t=1]}  (page 31)
        if len(bars) == 0 and num_ticks >= expected_num_ticks_init:
            expected_imbalance = max(
                expected_num_ticks * max(expected_p_buy, 1 - expected_p_buy),
                1e-6
            )

        # Tick Run Bar Stopping Definition — page 31
        # T* = arg min_T { θ_T >= E0[T] * max{P[b_t=1], 1 - P[b_t=1]} }
        # Close the bar when the run length exceeds the expected run length
        if expected_imbalance != 0 and theta >= expected_imbalance:
            open_p  = collector[0]
            high_p  = np.max(collector)
            low_p   = np.min(collector)
            close_p = collector[-1]
            # VWAP = Σ(price * volume) / Σ(volume)
            vwap    = vol_price / cumm_vol

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))
            bar_lengths.append(num_ticks)

            # Track proportion of buy ticks for EWMA update of P[b_t=1]
            buy_proportion = pos_run / num_ticks if num_ticks > 0 else 0.5
            buy_tick_proportions.append(buy_proportion)

            # Reset accumulators
            pos_run = 0
            neg_run = 0
            cumm_vol = 0
            vol_price = 0
            collector = []
            num_ticks = 0

            # Tick Run Bar Expected Imbalance update — page 31
            # E0[θ_T] = E0[T] * max{P[b_t=1], 1 - P[b_t=1]}
            # Update E0[T] via EWMA of bar lengths
            # Update P[b_t=1] via EWMA of buy tick proportions from prior bars
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


def volume_run_bars(df, expected_num_ticks_init=10, num_prev_bars=3):
    # Volume/Dollar Run Bars (VRBs/DRBs) — AFML Chapter 2, page 32
    # Extends tick run bars by accumulating volume on each side instead of tick counts.
    pos_run = 0   # accumulated buy volume in current bar
    neg_run = 0   # accumulated sell volume in current bar
    cumm_vol = 0
    vol_price = 0
    collector = []
    bars = []

    bar_lengths = []
    buy_vol_proportions = []   # tracks E0[v_t | b_t=1] per bar
    sell_vol_proportions = []  # tracks E0[v_t | b_t=-1] per bar
    num_ticks = 0

    expected_num_ticks = expected_num_ticks_init
    expected_p_buy = 0.5
    expected_buy_vol = 0.01    # initial guess for avg buy volume per tick
    expected_sell_vol = 0.01   # initial guess for avg sell volume per tick
    expected_imbalance = 0

    for i, (label, price, date, volume) in enumerate(zip(df['Label'], df['Price'], df['Date'], df['Volume'])):

        # Volume/Dollar Run Bar Length Definition — page 32
        # θ_T = max( Σ b_t*v_t where b_t=1, -Σ b_t*v_t where b_t=-1 )
        # Accumulate volume on each side independently — never reset mid-bar
        if label == 1:
            pos_run += volume   # add buy volume
        elif label == -1:
            neg_run += volume   # add sell volume

        # θ_T = max of both sides
        theta = max(pos_run, neg_run)

        cumm_vol += volume
        vol_price += price * volume
        collector.append(price)
        num_ticks += 1

        # Initialize expected imbalance using warmup period
        # E0[θ_T] = E0[T] * max{P[b_t=1]*E0[v_t|b_t=1], (1-P[b_t=1])*E0[v_t|b_t=-1]}
        if len(bars) == 0 and num_ticks >= expected_num_ticks_init:
            expected_imbalance = max(
                expected_num_ticks * max(
                    expected_p_buy * expected_buy_vol,
                    (1 - expected_p_buy) * expected_sell_vol
                ),
                1e-6
            )

        # Volume/Dollar Run Bar Stopping Rule — page 32
        # T* = arg min_T { θ_T >= E0[T] * max{P[b_t=1]*E0[v_t|b_t=1],
        #                                      (1-P[b_t=1])*E0[v_t|b_t=-1]} }
        # Close the bar when the volume run exceeds the expected volume run
        if expected_imbalance != 0 and theta >= expected_imbalance:
            open_p  = collector[0]
            high_p  = np.max(collector)
            low_p   = np.min(collector)
            close_p = collector[-1]
            # VWAP = Σ(price * volume) / Σ(volume)
            vwap    = vol_price / cumm_vol

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))
            bar_lengths.append(num_ticks)

            # Track avg buy/sell volume per tick for EWMA updates
            buy_vol_proportions.append(pos_run / num_ticks if num_ticks > 0 else expected_buy_vol)
            sell_vol_proportions.append(neg_run / num_ticks if num_ticks > 0 else expected_sell_vol)

            # Reset accumulators
            pos_run = 0
            neg_run = 0
            cumm_vol = 0
            vol_price = 0
            collector = []
            num_ticks = 0

            # Volume/Dollar Run Bar Expected Imbalance update — page 32
            # E0[θ_T] = E0[T] * max{P[b_t=1]*E0[v_t|b_t=1], (1-P[b_t=1])*E0[v_t|b_t=-1]}
            # Update all four components via EWMA of prior bar values
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
