import numpy as np
import pandas as pd

def etf_trick(open_prices, close_prices, alloc_weights, point_values, dividends, rebalance_dates, trans_costs=None):
    """
    Implements the ETF Trick from AFML Chapter 2, Section 2.4.1.
    
    Converts a basket of instruments into a single $1 investment series.

    Parameters:
    - open_prices:      DataFrame, shape (T, I) — open price of each instrument at each bar
    - close_prices:     DataFrame, shape (T, I) — close price of each instrument at each bar
    - alloc_weights:    DataFrame, shape (T, I) — allocation weights ω_i,t for each instrument
    - point_values:     DataFrame, shape (T, I) — USD value of one point φ_i,t
    - dividends:        DataFrame, shape (T, I) — dividends/carry d_i,t
    - rebalance_dates:  list of bar indices where rebalancing occurs (set B)
    - trans_costs:      optional Series, shape (I,) — transaction cost per $1 traded τ_i

    Returns:
    - DataFrame with columns: K (portfolio value), rebalance_cost, bid_ask_cost, volume
    """

    T, I = close_prices.shape
    bars = close_prices.index

    # Initialize outputs
    K = pd.Series(index=bars, dtype=float)      # portfolio value
    h = pd.DataFrame(0.0, index=bars, columns=close_prices.columns)  # holdings
    rebalance_cost = pd.Series(0.0, index=bars) # c_t
    bid_ask_cost   = pd.Series(0.0, index=bars) # c~_t
    volume         = pd.Series(0.0, index=bars) # v_t

    if trans_costs is None:
        trans_costs = pd.Series(0.0, index=close_prices.columns)

    K.iloc[0] = 1.0  # K_0 = 1

    for t_idx in range(1, T):
        t     = bars[t_idx]
        t_prev = bars[t_idx - 1]

        # --- Compute holdings h_i,t ---
        # Formula: h_i,t = (ω_i,t * K_t) / (o_i,t+1 * φ_i,t * Σ|ω_i,t|)  if t ∈ B
        #                  h_i,t-1                                           otherwise
        if t_prev in rebalance_dates:
            w    = alloc_weights.loc[t_prev]
            denom = w.abs().sum()
            for i in close_prices.columns:
                h.loc[t, i] = (w[i] * K.iloc[t_idx - 1]) / (open_prices.loc[t, i] * point_values.loc[t_prev, i] * denom)
        else:
            h.loc[t] = h.loc[t_prev]

        # --- Compute price change δ_i,t ---
        # Formula: δ_i,t = p_i,t - o_i,t  if (t-1) ∈ B
        #                  Δp_i,t          otherwise
        if t_prev in rebalance_dates:
            delta = close_prices.loc[t] - open_prices.loc[t]
        else:
            delta = close_prices.loc[t] - close_prices.loc[t_prev]

        # --- Update portfolio value K_t ---
        # Formula: K_t = K_{t-1} + Σ h_{i,t-1} * φ_{i,t} * (δ_{i,t} + d_{i,t})
        pnl = (h.loc[t_prev] * point_values.loc[t] * (delta + dividends.loc[t])).sum()
        K.iloc[t_idx] = K.iloc[t_idx - 1] + pnl

        # --- Compute transaction costs (if rebalancing) ---
        if t_prev in rebalance_dates:
            # Rebalance cost: c_t = Σ (|h_{i,t-1}|*p_{i,t} + |h_{i,t}|*o_{i,t+1}) * φ_{i,t} * τ_i
            rebalance_cost.loc[t] = (
                (h.loc[t_prev].abs() * close_prices.loc[t] +
                 h.loc[t].abs() * open_prices.loc[t]) *
                point_values.loc[t] * trans_costs
            ).sum()

            # Bid-ask cost: c~_t = Σ |h_{i,t-1}| * p_{i,t} * φ_{i,t} * τ_i
            bid_ask_cost.loc[t] = (
                h.loc[t_prev].abs() * close_prices.loc[t] *
                point_values.loc[t] * trans_costs
            ).sum()

        # --- Compute tradeable volume ---
        # Formula: v_t = min_i { v_{i,t} / |h_{i,t-1}| }
        h_prev_abs = h.loc[t_prev].abs()
        if (h_prev_abs > 0).any():
            volume.loc[t] = (close_prices.loc[t] / h_prev_abs.replace(0, np.nan)).min()

    result = pd.DataFrame({
        'K':               K,
        'rebalance_cost':  rebalance_cost,
        'bid_ask_cost':    bid_ask_cost,
        'volume':          volume
    })

    return result