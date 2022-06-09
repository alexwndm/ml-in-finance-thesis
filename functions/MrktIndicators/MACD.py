# MACD, see Gerard Appel, 2005

import pandas as pd
from functions.EWMA import get_EMA


# MACD with envelope around signal line
def get_MACD_envelope(close, window=250, fast_t=12, slow_t=26, signal_t=9, streched=True, factor=None):
    if window is None:
        min_periods = None
        sd_window = 250
    else:
        min_periods = int(window / 10)
        sd_window = window
    if streched and window is not None:
        slow_t = window
        fast_t = fast_t / 26 * window
        signal_t = signal_t / 26 * window
    if not streched and factor is not None:
        slow_t *= factor
        fast_t *= factor
        signal_t *= factor

    macd = get_EMA(close, min_periods=min_periods, span=fast_t, lim_window=window) \
           - get_EMA(close, min_periods=min_periods, span=slow_t, lim_window=window)
    macd_std = macd.rolling(window=sd_window, min_periods=None).std()
    signal = get_EMA(macd, min_periods=min_periods, span=signal_t, lim_window=window)

    lower = signal - macd_std
    upper = signal + macd_std
    df0 = pd.concat([macd, lower, upper], axis=1)
    df0.columns = ["MACD", "Lower", "Upper"]
    return df0


