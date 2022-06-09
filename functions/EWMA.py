# Exp moving avergage with limited rolling window, fast

import numpy as np


# helper func, perform ewm for rolling in get_EMA, see pandas.ewm w/ adjusted=True
def perform_ema(close, span=10):
    alpha = 2 / (span + 1)
    scaling_factors = np.power(np.repeat(1. - alpha, close.shape[0]), range(close.shape[0])[::-1],
                               dtype=np.float64)
    return np.sum(np.multiply(scaling_factors, close)) / np.sum(scaling_factors)


# get exp weighted moving average with rolling window of limited size
def get_EMA(close, span, lim_window, min_periods=None):
    if lim_window is None:  # no restriction necessary, use standard EMA
        return close.ewm(span=span, adjust=False).mean()
    if min_periods is None:
        min_periods = int(lim_window/4)
    return close.rolling(lim_window, min_periods=min_periods).apply(perform_ema, args=(span,), raw=True)

