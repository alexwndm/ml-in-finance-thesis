# Envelope indicator
import pandas as pd


# return lower and upper envelope around mid
def get_envelope(close, window=250, min_periods=None, envelope_prct=0.03):
    if min_periods is None:
        min_periods = int(window * 0.5)
    close_mean = close.rolling(window, min_periods=min_periods).mean()
    lower = close_mean - envelope_prct * close_mean
    upper = close_mean + envelope_prct * close_mean
    df0 = pd.concat([close, lower, upper], axis=1)
    df0.columns = ["CloseAdj", "Lower", "Upper"]
    return df0
