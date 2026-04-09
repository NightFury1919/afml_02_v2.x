import numpy as np
import pandas as pd

def roll_gaps(series, dictio={'Instrument': 'Instrument', 'Open': 'Open', 'Close': 'Close'}, match_end=True):
    # Find the dates where the contract changes (roll dates)
    roll_dates = series[dictio['Instrument']].drop_duplicates(keep='first').index

    # Initialize gaps series as zeros
    gaps = series[dictio['Close']] * 0

    # Get integer positions of the days just before each roll
    iloc = list(series.index)
    iloc = [iloc.index(i) - 1 for i in roll_dates]

    # Gap at each roll = new contract open - previous contract close
    gaps.loc[roll_dates[1:]] = (
        series[dictio['Open']].loc[roll_dates[1:]].values -
        series[dictio['Close']].iloc[iloc[1:]].values
    )

    # Cumulative sum of gaps
    gaps = gaps.cumsum()

    # Roll backward: align end of series (match_end=True)
    # Roll forward: align start of series (match_end=False)
    if match_end:
        gaps -= gaps.iloc[-1]

    return gaps


def get_rolled_series(series, dictio={'Instrument': 'Instrument', 'Open': 'Open', 'Close': 'Close'}, match_end=True):
    # Compute the gaps at each roll date
    gaps = roll_gaps(series, dictio=dictio, match_end=match_end)

    # Subtract gaps from close and open prices to get smooth series
    rolled = series.copy(deep=True)
    for fld in [dictio['Open'], dictio['Close']]:
        rolled[fld] -= gaps

    return rolled


def non_negative_rolled_prices(series, dictio={'Instrument': 'Instrument', 'Open': 'Open', 'Close': 'Close'}, match_end=True):
    # Step 1: get rolled price series
    gaps = roll_gaps(series, dictio=dictio, match_end=match_end)
    rolled = series.copy(deep=True)
    for fld in [dictio['Open'], dictio['Close']]:
        rolled[fld] -= gaps

    # Step 2: compute returns as rolled price change / previous raw close
    rolled['Returns'] = rolled[dictio['Close']].diff() / series[dictio['Close']].shift(1)

    # Step 3: form $1 investment price series using cumulative returns
    rolled['rPrices'] = (1 + rolled['Returns']).cumprod()

    return rolled