# Donchian channel

import pandas as pd


# Donchian channel
def get_donchian_channel(close, window=17):
    roll_min = close.rolling(window, min_periods=2).min()
    roll_max = close.rolling(window, min_periods=2).max()
    df0 = pd.concat([close, roll_min, roll_max], axis=1)
    df0.columns = ["CloseAdj", "Lower", "Upper"]
    return df0


