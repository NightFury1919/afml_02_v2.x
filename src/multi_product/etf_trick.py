import numpy as np
import pandas as pd

def etf_trick(open_prices, close_prices, alloc_weights, point_values, dividends, rebalance_dates, trans_costs=None):
    # ETF Trick — AFML Chapter 2, Section 2.4.1, pages 33-34
    #
    # Converts a basket of multiple instruments into a single continuous
    # $1 investment series (K_t), eliminating issues with:
    #   - Changing weights causing artificial convergence
    #   - Negative spread values
    #   - Roll gaps in futures contracts
    #
    # Three core formulas from page 34:
    #
    # Holdings h_i,t:
    #   h_i,t = (ω_i,t * K_t) / (o_i,t+1 * φ_i,t * Σ|ω_i,t|)   if t ∈ B (rebalance)
    #   h_i,t = h_i,t-1                                            otherwise
    #
    # Price change δ_i,t:
    #   δ_i,t = p_i,t - o_i,t    if (t-1) ∈ B (first bar after rebalance)
    #   δ_i,t = Δp_i,t           otherwise (day-to-day change)
    #
    # Portfolio value K_t:
    #   K_t = K_{t-1} + Σ_i h_{i,t-1} * φ_{i,t} * (δ_{i,t} + d_{i,t})
    #   K_0 = 1 (start with $1)
    #
    # Three auxiliary outputs also from page 34:
    #   Rebalance cost: c_t  = Σ (|h_{i,t-1}|*p_{i,t} + |h_{i,t}|*o_{i,t+1}) * φ_{i,t} * τ_i
    #   Bid-ask cost:   c~_t = Σ |h_{i,t-1}| * p_{i,t} * φ_{i,t} * τ_i
    #   Volume:         v_t  = min_i { v_{i,t} / |h_{i,t-1}| }

    T, I = close_prices.shape
    bars = close_prices.index

    # Initialize outputs
    K = pd.Series(index=bars, dtype=float)      # K_t: portfolio value
    h = pd.DataFrame(0.0, index=bars, columns=close_prices.columns)  # h_i,t: holdings
    rebalance_cost = pd.Series(0.0, index=bars) # c_t: rebalance cost
    bid_ask_cost   = pd.Series(0.0, index=bars) # c~_t: bid-ask cost
    volume         = pd.Series(0.0, index=bars) # v_t: tradeable volume

    if trans_costs is None:
        trans_costs = pd.Series(0.0, index=close_prices.columns)

    K.iloc[0] = 1.0  # K_0 = 1 (initial $1 investment)

    for t_idx in range(1, T):
        t      = bars[t_idx]
        t_prev = bars[t_idx - 1]

        # --- Holdings h_i,t (page 34) ---
        # On rebalance dates: recalculate based on current portfolio value K
        # and target allocation weights ω, entering at next open price.
        # On other dates: carry forward previous holdings unchanged.
        if t_prev in rebalance_dates:
            w     = alloc_weights.loc[t_prev]
            denom = w.abs().sum()  # Σ|ω_i,t| — de-levers the allocations
            for i in close_prices.columns:
                # h_i,t = (ω_i,t * K_t) / (o_i,t+1 * φ_i,t * Σ|ω_i,t|)
                h.loc[t, i] = (w[i] * K.iloc[t_idx - 1]) / (
                    open_prices.loc[t, i] * point_values.loc[t_prev, i] * denom
                )
        else:
            h.loc[t] = h.loc[t_prev]  # h_i,t = h_i,t-1

        # --- Price change δ_i,t (page 34) ---
        # First bar after rebalance: use close minus open (entered at open)
        # All other bars: use day-to-day close change
        if t_prev in rebalance_dates:
            delta = close_prices.loc[t] - open_prices.loc[t]   # p_i,t - o_i,t
        else:
            delta = close_prices.loc[t] - close_prices.loc[t_prev]  # Δp_i,t

        # --- Portfolio value K_t (page 34) ---
        # K_t = K_{t-1} + Σ_i h_{i,t-1} * φ_{i,t} * (δ_{i,t} + d_{i,t})
        pnl = (h.loc[t_prev] * point_values.loc[t] * (delta + dividends.loc[t])).sum()
        K.iloc[t_idx] = K.iloc[t_idx - 1] + pnl

        # --- Transaction costs — only computed on rebalance dates (page 34) ---
        if t_prev in rebalance_dates:
            # Rebalance cost c_t: cost of closing old positions and opening new ones
            # c_t = Σ (|h_{i,t-1}|*p_{i,t} + |h_{i,t}|*o_{i,t+1}) * φ_{i,t} * τ_i
            rebalance_cost.loc[t] = (
                (h.loc[t_prev].abs() * close_prices.loc[t] +
                 h.loc[t].abs() * open_prices.loc[t]) *
                point_values.loc[t] * trans_costs
            ).sum()

            # Bid-ask cost c~_t: cost of buying/selling one unit of the virtual ETF
            # c~_t = Σ |h_{i,t-1}| * p_{i,t} * φ_{i,t} * τ_i
            bid_ask_cost.loc[t] = (
                h.loc[t_prev].abs() * close_prices.loc[t] *
                point_values.loc[t] * trans_costs
            ).sum()

        # --- Tradeable volume v_t (page 34) ---
        # v_t = min_i { v_{i,t} / |h_{i,t-1}| }
        # Limited by the least liquid instrument in the basket
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
