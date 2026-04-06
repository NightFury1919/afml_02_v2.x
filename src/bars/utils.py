import numpy as np
import pandas as pd

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

def tick_rule(df):
    b = np.ones(len(df['Price']))
    for i, delta in enumerate(df['Delta']):
        if i > 0:
            # Tick rule:
            if delta == 0:
                b[i] = b[i-1]
            else:
                b[i] = abs(delta) / delta
    df['Label'] = b
    return df

def estimate_buy_sell_probs(df):
    prob = pd.DataFrame(pd.pivot_table(df, index='Label', values='Price', aggfunc='count'))
    prob = np.array(prob)
    p_b = prob[1]/(prob[0]+prob[1])
    p_s = prob[0]/(prob[0]+prob[1])
    return p_b, p_s