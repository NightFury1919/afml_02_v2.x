import numpy as np
import pandas as pd

def roll_gaps(series, dictio={'Instrument': 'Instrument', 'Open': 'Open', 'Close': 'Close'}, match_end=True):
    # Roll Gaps — AFML Chapter 2, Section 2.4.3, page 37
    # (Part of Snippet 2.2 in the book)
    #
    # Computes the cumulative price gaps introduced at each futures contract roll.
    # When a contract expires and a new one begins, the price may jump artificially.
    # This function identifies those gaps and returns their cumulative sum,
    # which can then be subtracted from the raw price series.
    #
    # Formula (page 37):
    #   gap_t = o_{new contract, t} - p_{old contract, t-1}   at each roll date
    #   gaps  = cumsum(gap_t)
    #
    # match_end=True:  backward roll — aligns price at the END of the series
    # match_end=False: forward roll  — aligns price at the START of the series

    # Find dates where the contract identifier changes (roll dates)
    roll_dates = series[dictio['Instrument']].drop_duplicates(keep='first').index

    # Initialize gaps as zero for all dates
    gaps = series[dictio['Close']] * 0

    # Get integer positions of the day just before each roll date
    iloc = list(series.index)
    iloc = [iloc.index(i) - 1 for i in roll_dates]

    # gap at each roll = new contract open - previous contract close
    gaps.loc[roll_dates[1:]] = (
        series[dictio['Open']].loc[roll_dates[1:]].values -
        series[dictio['Close']].iloc[iloc[1:]].values
    )

    # Cumulative sum of all gaps over time
    gaps = gaps.cumsum()

    # Backward roll: subtract the final cumulative gap so the series
    # aligns at the end (most recent prices match raw prices)
    if match_end:
        gaps -= gaps.iloc[-1]

    return gaps


def get_rolled_series(series, dictio={'Instrument': 'Instrument', 'Open': 'Open', 'Close': 'Close'}, match_end=True):
    # Rolled Price Series — AFML Chapter 2, Section 2.4.3, page 37
    # (Snippet 2.2 in the book)
    #
    # Applies the roll gap correction to produce a smooth continuous price series
    # by subtracting the cumulative gaps from open and close prices.
    # The resulting series has no artificial jumps at roll dates.

    # Compute cumulative gaps at each roll date
    gaps = roll_gaps(series, dictio=dictio, match_end=match_end)

    # Subtract gaps from both open and close prices
    rolled = series.copy(deep=True)
    for fld in [dictio['Open'], dictio['Close']]:
        rolled[fld] -= gaps

    return rolled


def non_negative_rolled_prices(series, dictio={'Instrument': 'Instrument', 'Open': 'Open', 'Close': 'Close'}, match_end=True):
    # Non-Negative Rolled Price Series — AFML Chapter 2, Section 2.4.3, page 37
    # (Snippet 2.3 in the book)
    #
    # Converts the rolled price series into a $1 investment series using
    # cumulative returns, guaranteeing the series is always positive.
    # Raw rolled prices can go negative in contango markets (e.g. natural gas).
    #
    # Three-step process from page 37:
    #   Step 1: Compute rolled prices (subtract cumulative gaps)
    #   Step 2: r_t = rolled_close_t / raw_close_{t-1} - 1  (percentage return)
    #   Step 3: rPrices = (1 + r).cumprod()  ($1 compounded by daily returns)

    # Step 1: Get rolled price series (gaps removed)
    gaps = roll_gaps(series, dictio=dictio, match_end=match_end)
    rolled = series.copy(deep=True)
    for fld in [dictio['Open'], dictio['Close']]:
        rolled[fld] -= gaps

    # Step 2: Compute returns using rolled close / previous raw close
    # Using raw close in denominator prevents the gap from affecting return magnitude
    rolled['Returns'] = rolled[dictio['Close']].diff() / series[dictio['Close']].shift(1)

    # Step 3: Compound returns into a $1 investment series
    # rPrices_t = Π_{s=1}^{t} (1 + r_s)
    rolled['rPrices'] = (1 + rolled['Returns']).cumprod()

    return rolled
