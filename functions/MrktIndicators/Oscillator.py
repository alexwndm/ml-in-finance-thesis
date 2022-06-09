# Trend oscillator by Stanislaus Maier-Paape

import numpy as np
import pandas as pd

from functions.EWMA import get_EMA


# gradient oscillator envelope
def get_osci_envelope(close, window=100, osci_period=5, alpha=7, vola_window=1000, time_restriction=True):
    min_periods = int(window / 4)

    if time_restriction:
        close_ma = get_EMA(close, span=window, lim_window=None, min_periods=min_periods)
        # close_ma = get_EMA(close, span=window, lim_window=window, min_periods=min_periods)
    else:
        close_ma = close.rolling(window=window, min_periods=min_periods).mean()

    grad = (close_ma - close_ma.shift(1)) / close
    if time_restriction:
        grad_ma = get_EMA(grad, span=2*osci_period+1, lim_window=None, min_periods=min_periods)
        # grad_ma = get_EMA(grad, span=2*osci_period+1, lim_window=window, min_periods=min_periods)
    else:
        grad_ma = grad.rolling(window=osci_period).mean()
    osci = 0.5 + np.arctan(grad_ma * 200) / np.pi

    vola = (osci - 0.5) ** 2
    if time_restriction:
        vola_ma = vola.rolling(window=min(window, vola_window), min_periods=min_periods).mean()
    else:
        vola_ma = get_EMA(vola, span=vola_window, lim_window=None)

    lower = 0.5 - alpha / 10 * np.sqrt(vola_ma)
    upper = 0.5 + alpha / 10 * np.sqrt(vola_ma)
    df0 = pd.concat([osci, lower, upper], axis=1)
    df0.columns = ["Oscillator", "Lower", "Upper"]
    return df0

