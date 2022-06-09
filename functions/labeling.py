# functions to label data

# general
import pandas as pd
import numpy as np


# get label, fixed time horizon
def get_label_fixed_time_horizon(close, horizon=10, returns_thresh=0.03):
    returns = close.shift(-horizon) / close - 1
    label = np.zeros(returns.shape[0]) + (returns > returns_thresh) - (returns < -returns_thresh)
    label[-horizon:] = np.nan
    return label


# get label, wrapper for triple-barrier method
def get_label_triple_barrier(close, profit_taking=True, stop_loss=True, horizon=10,
                             target=None, returns_thresh=0.03):
    # print("Adding triple-barrier label.")
    if target is None:
        target = np.zeros(close.shape[0]) + returns_thresh
    events_ = pd.DataFrame({"t1": close.reset_index()["Date"].shift(-horizon).array,
                            "trgt": target,
                            "side": np.ones(close.shape[0], dtype=float)}, index=close.index)
    out = applyPtSlOnT1(close=close, events=events_,
                        ptSl=[profit_taking * 1, stop_loss * 1],
                        molecule=np.ones(events_.shape[0], dtype=bool))
    # get first touch, sometimes both horizontal barriers are hit
    first_touch = out.idxmin(axis=1)
    label = -1 * (first_touch == "sl") + 1 * (first_touch == "pt")
    # label = 0 * (first_touch == "sl") + 1 * (first_touch == "pt") \
    #         + 1 * ((close.shift(-horizon) / close - 1) > 0) * (first_touch == "t1")
    return label


# See Advances in Financial ML, Lopez, p. 45
def applyPtSlOnT1(close, events, ptSl, molecule):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]  # for parallelization later
    out = events_[['t1']].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']  # profit taking limit
    else:
        pt = pd.Series(index=events.index)  # NaNs
    if ptSl[1] > 0:
        sl = -ptSl[1] * events_['trgt']  # stop loss limit
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1]  # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # path returns
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # earliest stop loss.
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # earliest profit taking.
    return out


# helper functions for Edge Ratio label
# get Maximum Favorable Excursion
def get_MFE(close):
    close = close - close[0]
    return np.max(close)


# get Maximum Adverse Excursion
def get_MAE(close):
    close = close - close[0]
    return np.abs(np.min(close))


# get label, edge ratio. ATR required
# See Way of the Turtle, Faith, 2007, p. 67
def get_label_edge_ratio(close, atr=None, horizon=10, min_eratio=1):
    # vola measure not necessary for labelling, save time
    # if atr is None:  # use vola instead
    #     logRets = np.log(close / close.shift(1))
    #     atr = logRets.rolling(window=21, min_periods=3).std()
    # Maximum Favorable Excursion
    MFE = close.rolling(window=horizon, min_periods=horizon).apply(get_MFE, raw=True)
    # MFE = MFE / atr.shift(horizon)  # standardizing not necessary for labelling
    # Maximum Adverse Excursion
    MAE = close.rolling(window=horizon, min_periods=horizon).apply(get_MAE, raw=True)
    # MAE = MAE / atr.shift(horizon)  # standardizing not necessary for labelling
    # Edge Ratio
    ERatio = MFE / MAE
    # ERatio = ERatio.replace(np.inf, np.max(ERatio[np.isfinite(ERatio)]))
    return ((ERatio > min_eratio) * 1).shift(-horizon)


# get label, 1 for high Edge Ratio and pos returns, -1 for low ER or neg rets
def get_label_edge_ratio_return(close, atr=None, horizon=10, min_eratio=1, min_ret=0):
    return_label = (close.shift(-horizon) / close - 1 > min_ret) * 1  # tail should be NaN, but okay bc ER_label tail is NaN
    edge_ratio_label = get_label_edge_ratio(close, atr=atr, horizon=horizon, min_eratio=min_eratio)
    return 2 * return_label * edge_ratio_label - 1  # -1 or 1
