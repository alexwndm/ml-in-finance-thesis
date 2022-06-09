
import numpy as np
import pandas as pd


# trend oscillator
# 2 bull, 1 bull cor, 0 undefined, -1 bear cor, -2 bear
def get_phase(close, window=200, bull_cor=-0.1, bear=-0.2):
    # set thresholds
    bear_to_bear_cor = 1 / (1 + bull_cor)
    bear_to_bull = 1 / (1 + bear)
    bull_to_bull_cor = 1 + bull_cor
    bull_to_bear = 1 + bear

    # get min/max
    roll_min = close.rolling(window, min_periods=2).min()
    roll_max = close.rolling(window, min_periods=2).max()

    # phase indicator
    phase = pd.Series(np.zeros(close.shape[0]), index=close.index)
    if close.iloc[1] >= close.iloc[0]:
        phase.iloc[0] = 2
        phase.iloc[1] = 2
    else:
        phase.iloc[0] = -2
        phase.iloc[1] = -2
    # loop through close values
    for t in range(2, phase.shape[0]):
        ratio_high = close.iloc[t] / roll_max.iloc[t - 1]
        ratio_low = close.iloc[t] / roll_min.iloc[t - 1]
        curr_phase = 0
        if ratio_high >= 1:  # new high
            curr_phase = 2
        elif ratio_low <= 1:  # new low
            curr_phase = -2
        # if bearish, cor?
        elif phase.iloc[t - 1] < 0:
            if ratio_low >= bear_to_bull:
                curr_phase = 2
            elif ratio_low >= bear_to_bear_cor:
                curr_phase = -1
            else:
                curr_phase = -2
        # if bullish, cor?
        elif phase.iloc[t - 1] > 0:
            if ratio_high <= bull_to_bear:
                curr_phase = -2
            elif ratio_high <= bull_to_bull_cor:
                curr_phase = 1
            else:
                curr_phase = 2
        phase.iloc[t] = curr_phase
    return phase


# continous trend oscillator
# bullish in [1,3], bearish in [-3,-1]
def get_cont_phase(close, window=200, bear=-0.2):
    # set thresholds
    bear_to_bull = 1 / (1 + bear)
    bull_to_bear = 1 + bear

    # get min/max
    roll_min = close.rolling(window, min_periods=2).min()
    roll_max = close.rolling(window, min_periods=2).max()

    # phase indicator
    phase = pd.Series(np.zeros(close.shape[0]), index=close.index)
    if close.iloc[1] >= close.iloc[0]:
        phase.iloc[0] = 3
        phase.iloc[1] = 3
    else:
        phase.iloc[0] = -3
        phase.iloc[1] = -3
    # loop through close values
    for t in range(2, phase.shape[0]):
        ratio_high = close.iloc[t] / roll_max.iloc[t - 1]
        ratio_low = close.iloc[t] / roll_min.iloc[t - 1]
        curr_phase = 0
        if ratio_high >= 1:  # new high
            curr_phase = 3
        elif ratio_low <= 1:  # new low
            curr_phase = -3
        # if bearish, cor?
        elif phase.iloc[t - 1] < 0:
            if ratio_low >= bear_to_bull:
                curr_phase = 3
            else:
                curr_phase = - 2 / (bear_to_bull - 1) * (1 - ratio_low) - 3  # in [-3,-1]
        # if bullish, cor?
        elif phase.iloc[t - 1] > 0:
            if ratio_high <= bull_to_bear:
                curr_phase = -3
            else:
                curr_phase = 2 / (bull_to_bear - 1) * (1 - ratio_high) + 3  # in [1,3]
        phase.iloc[t] = curr_phase
    return phase


# continous trend oscillator
# bullish in [4,5],, bull correction in [1,3], bearish in [-5,-4], bear cor in [-3,-1]
def get_cont_phase2(close, window=200, bull_cor=-0.1, bear=-0.2):
    # set thresholds
    bear_to_bear_cor = 1 / (1 + bull_cor)
    bear_to_bull = 1 / (1 + bear)
    bull_to_bull_cor = 1 + bull_cor
    bull_to_bear = 1 + bear

    # get min/max
    roll_min = close.rolling(window, min_periods=2).min()
    roll_max = close.rolling(window, min_periods=2).max()

    # set start
    phase = pd.Series(np.zeros(close.shape[0]), index=close.index)
    if close.iloc[1] >= close.iloc[0]:
        phase.iloc[0] = 5
        phase.iloc[1] = 5
    else:
        phase.iloc[0] = -5
        phase.iloc[1] = -5

    # loop through close values
    for t in range(2, phase.shape[0]):
        ratio_high = close.iloc[t] / roll_max.iloc[t - 1]
        ratio_low = close.iloc[t] / roll_min.iloc[t - 1]
        curr_phase = 0
        if ratio_high >= 1:  # new high
            curr_phase = 5
        elif ratio_low <= 1:  # new low
            curr_phase = -5
        # bearish
        elif -5 <= phase.iloc[t - 1] <= -4:
            if ratio_low >= bear_to_bull:
                curr_phase = 5
            elif ratio_low >= bear_to_bear_cor:
                curr_phase = -2
            else:
                curr_phase = - 1 / (bear_to_bear_cor - 1) * (1 - ratio_low) - 5  # in [-5,-4]
        # bear correction
        elif -3 <= phase.iloc[t - 1] <= -1:
            if ratio_low >= bear_to_bull:
                curr_phase = 5
            else:
                curr_phase = - 2 / (bear_to_bull - 1) * (1 - ratio_low) - 3  # in [-3,-1]
        # bullish
        elif 4 <= phase.iloc[t - 1] <= 5:
            if ratio_high <= bull_to_bear:
                curr_phase = -5
            elif ratio_high <= bull_to_bull_cor:
                curr_phase = 2
            else:
                curr_phase = 1 / (bull_to_bear - 1) * (1 - ratio_high) + 5  # in [4,5]
        # bull correction
        elif 1 <= phase.iloc[t - 1] <= 3:
            if ratio_high <= bull_to_bear:
                curr_phase = -5
            else:
                curr_phase = 2 / (bull_to_bear - 1) * (1 - ratio_high) + 3  # in [1,3]
        phase.iloc[t] = curr_phase
    return phase
