# Bollinger Band

import numpy as np
import pandas as pd


def get_bollinger_band(close, window=250, min_periods=None, std_factor=2):
    if window is None:
        window = close.shape[0]
    if min_periods is None:
        min_periods = int(window * 0.5)

    close_mean = close.rolling(window, min_periods=min_periods).mean()
    close_std = close.rolling(window, min_periods=min_periods).std()

    upper = close_mean + std_factor * close_std
    lower = close_mean - std_factor * close_std
    df0 = pd.concat([close, lower, upper], axis=1)
    df0.columns = ["CloseAdj", "Lower", "Upper"]
    return df0


# Bollinger Band by Daniel Haase
def get_trend_bollinger_Haase(close, window=250, std_factor=2):
    if window is None:
        window = close.shape[0]

    close_mean = close.rolling(window, min_periods=1).mean()
    close_std = close.rolling(window, min_periods=1).std()
    close_std.iloc[0] = 0  # no information available
    upper = close_mean + std_factor * close_std
    lower = close_mean - std_factor * close_std
    limit = -1  # to be replaced in first loop
    stop = 0

    invested = pd.Series(np.zeros(len(close)), index=close.index, name="Trend" + str(window) + "Days")
    invested.iloc[0] = 0  # start not invested
    for i in range(1, close.shape[0]):
        # if not invested
        if not invested.iloc[i-1]:
            if upper.iloc[i] < limit or limit == -1:
                limit = upper.iloc[i]  # update limit
            if close.iloc[i] >= limit:  # passed profit taking limit
                invested.iloc[i] = 1
                stop = lower.iloc[i]  # new stop
        # if invested
        else:
            if lower.iloc[i] > stop:
                stop = lower.iloc[i]  # update stop
            if close.iloc[i] <= stop:
                invested.iloc[i] = 0
                limit = upper.iloc[i]
            else:
                invested.iloc[i] = 1  # stay invested
    return invested


# Bollinger Band induced trend
def get_trend_bollinger3(close, window=250, std_factor=2, rolling_func=["None", "None"]):
    if window is None:
        window = close.shape[0]

    close_mean = close.rolling(window, min_periods=int(window * 0.1)).mean()
    close_std = close.rolling(window, min_periods=int(window * 0.1)).std()
    upper = close_mean + std_factor * close_std
    lower = close_mean - std_factor * close_std

    if rolling_func[0] == "min":  # rolling min for lower band
        lower = lower.rolling(window, min_periods=int(window * 0.1)).min()
    elif rolling_func[0] == "max":  # max
        lower = lower.rolling(window, min_periods=int(window * 0.1)).max()
    if rolling_func[1] == "min":  # rolling min for upper band
        upper = upper.rolling(window, min_periods=int(window * 0.1)).min()
    elif rolling_func[1] == "max":  # max
        upper = upper.rolling(window, min_periods=int(window * 0.1)).max()

    hit_high = (close >= upper)
    hit_low = (close <= lower)

    trend = pd.Series(np.zeros(len(close)), index=close.index, name="Trend" + str(window) + "Days")
    # trend.iloc[0] = 1  # start with up trend w/o any knowledge
    for i in range(1, len(close)):
        if hit_low[i] or hit_high[i]:  # only one can be true
            trend[i] = 2 * hit_high[i] - 1  # 1 if high, else -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]
    return trend


