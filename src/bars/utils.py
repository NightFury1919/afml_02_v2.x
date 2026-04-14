import numpy as np
import pandas as pd

def ewma(arr, window):
    # Exponentially Weighted Moving Average (EWMA)
    # Used throughout chapter 2 to adaptively update expected number of ticks
    # and expected imbalance after each bar closes. Referenced on pages 31-32.
    # Formula: ewma_t = alpha * x_t + (1 - alpha) * ewma_{t-1}
    # where alpha = 2 / (window + 1)
    if len(arr) == 0:
        return 0

    alpha = 2 / (window + 1)
    ewma_val = arr[0]

    for i in range(1, len(arr)):
        ewma_val = alpha * arr[i] + (1 - alpha) * ewma_val

    return ewma_val

def delta(df):
    # Computes Δp_t — the price change between consecutive ticks.
    # Used as input to the Tick Rule (page 29).
    a = np.diff(df['Price'])
    a = np.insert(a, 0, 0)
    df['Delta'] = a
    return df

def tick_rule(df):
    # Tick Rule — AFML Chapter 2, page 29
    # Assigns a direction b_t to each trade:
    #   b_t = b_{t-1}         if Δp_t = 0  (price unchanged, carry forward)
    #   b_t = |Δp_t| / Δp_t  if Δp_t ≠ 0  (gives +1 for uptick, -1 for downtick)
    # b_t is used as a proxy for trade direction (buy = +1, sell = -1)
    b = np.ones(len(df['Price']))
    for i, delta in enumerate(df['Delta']):
        if i > 0:
            if delta == 0:
                # Carry forward previous direction
                b[i] = b[i-1]
            else:
                # +1 if price went up, -1 if price went down
                b[i] = abs(delta) / delta
    df['Label'] = b
    return df

def estimate_buy_sell_probs(df):
    # Estimates p_b and p_s — the probability that a tick is a buy or sell.
    # Used to initialize expected imbalance before the first bar is formed.
    # p_b = count of buy ticks / total ticks
    # p_s = count of sell ticks / total ticks
    # Referenced on page 31 as initial conditions for imbalance bars.
    prob = pd.DataFrame(pd.pivot_table(df, index='Label', values='Price', aggfunc='count'))
    prob = np.array(prob)
    p_b = prob[1]/(prob[0]+prob[1])
    p_s = prob[0]/(prob[0]+prob[1])
    return p_b, p_s
