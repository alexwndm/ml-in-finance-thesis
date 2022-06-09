# %% initialization ====================================================================================================

# general
import pandas as pd
import numpy as np

# other
from functions.labeling import get_label_fixed_time_horizon, get_label_triple_barrier, \
    get_label_edge_ratio, get_label_edge_ratio_return
from functions.MrktIndicators.Envelope import get_envelope
from functions.MrktIndicators.Bollinger import get_bollinger_band
from functions.MrktIndicators.Donchian import get_donchian_channel
from functions.MrktIndicators.MACD import get_MACD_envelope
from functions.MrktIndicators.Oscillator import get_osci_envelope


# add features and labels to dataframe
def add_features(data, return_horizons=[1, 10], rolling_windows=None, label_horizons=[10],
                 value="CloseAdj", label_type="triple_barrier", vola_weights="equal_weights", is_index=False):
    new_data = []
    counter = 1
    for df in data:
        print("Working on asset %d of %d." % (counter, len(data)))
        counter += 1

        # add returns
        if 1 not in return_horizons and not is_index:  # for backtesting
            df["Return" + str(1)] = df[value] / df[value].shift(1) - 1
        for window in return_horizons:
            # add previous returns
            df["Return" + str(window)] = df[value] / df[value].shift(window) - 1

        if is_index:
            new_data.append(df)
            continue  # rather than using Index- features, use Mrkt- later

        # add time features
        df["DayOfYear"] = df.index.dayofyear
        # df["DayOfMonth"] = df.index.day
        # df["DayOfWeek"] = df.index.dayofweek
        df["Year"] = df.index.year

        if rolling_windows is None:
            rolling_windows = return_horizons

        for window in rolling_windows:
            # add volatility features ----------------------------------------------------------------------------------
            # add vola
            if vola_weights == "ewma":
                df["VolaEWMA" + str(window)] = get_vola(close=df[value], window=window, weights="ewma")
            else:
                df["Vola" + str(window)] = get_vola(close=df[value], window=window, weights="equal_weights")
            # add DD
            df["DD" + str(window)] = get_roll_DD(df[value], window=window)

            # standardize close price ----------------------------------------------------------------------------------
            # add envelope
            envelope = get_envelope(df[value], window=window, envelope_prct=0.03)
            df["Envelope" + str(window)] = get_relative_trend(envelope)
            # df["EnvelopeTrend" + str(window)] = get_discrete_trend(envelope)
            # add Bollinger Band
            boll = get_bollinger_band(df[value], window=window, std_factor=2)
            df["Bollinger" + str(window)] = get_relative_trend(boll)
            # df["BollingerTrend" + str(window)] = get_discrete_trend(boll)
            # add Donchian channel
            donchian = get_donchian_channel(df[value], window=window)
            df["Donchian" + str(window)] = get_relative_trend(donchian)
            df["DonchianTrend" + str(window)] = get_discrete_trend(donchian)
            # add momentum features ------------------------------------------------------------------------------------
            # add MACD
            macd = get_MACD_envelope(df[value], window=window, streched=True)
            df["MACD" + str(window)] = get_relative_trend(macd)
            # df["MACDTrend" + str(window)] = get_discrete_trend(macd)
            # add gradient oscillator
            osci = get_osci_envelope(df[value], window=window)
            df["Oscillator" + str(window)] = get_relative_trend(osci)
            # df["OscillatorTrend" + str(window)] = get_discrete_trend(osci)

            # add adaptive envelopes
            # for f1, F1 in zip(["none", "min", "max"], ["None", "Min", "Max"]):
            for f1, F1 in zip(["min"], ["Min"]):
                # for f2, F2 in zip(["none", "min", "max"], ["None", "Min", "Max"]):
                for f2, F2 in zip(["max"], ["Max"]):
                    envelope_adapt = get_adaptive_envelope(envelope, window=window, lower_func=f1, upper_func=f2)
                    df["Env" + F1 + F2 + "Trend" + str(window)] = get_discrete_trend(envelope_adapt)

                    boll_adapt = get_adaptive_envelope(boll, window=window, lower_func=f1, upper_func=f2)
                    df["Boll" + F1 + F2 + "Trend" + str(window)] = get_discrete_trend(boll_adapt)

                    macd_adapt = get_adaptive_envelope(macd, window=window, lower_func=f1, upper_func=f2)
                    df["MACD" + F1 + F2 + "Trend" + str(window)] = get_discrete_trend(macd_adapt)

                    osci_adapt = get_adaptive_envelope(osci, window=window, lower_func=f1, upper_func=f2)
                    df["Osci" + F1 + F2 + "Trend" + str(window)] = get_discrete_trend(osci_adapt)

        # add label ----------------------------------------------------------------------------------------------------
        if label_horizons is not None:
            for horizon in label_horizons:
                if label_type == "triple_barrier":
                    df["Label" + str(horizon)] = get_label_triple_barrier(close=df[value], horizon=horizon,
                                                                          target=df["Vola"])
                elif label_type == "edge_ratio":
                    df["Label" + str(horizon)] = get_label_edge_ratio(close=df[value], atr=None, horizon=horizon)
                elif label_type == "edge_ratio_return":
                    df["Label" + str(horizon)] = get_label_edge_ratio_return(close=df[value], atr=None,
                                                                             horizon=horizon)
                else:  # fixed time horizon return
                    df["Label" + str(horizon)] = get_label_fixed_time_horizon(close=df[value], horizon=horizon,
                                                                              returns_thresh=0.03)
        if df.index[-1] < pd.to_datetime("2020-06-01"):  # asset delisted/bancrupt etc
            df_end = df.loc[df.index > pd.to_datetime("2020-01-01")]  # keep NAs at end
            df.loc[:, [label for label in df.columns if label[:5] == "Label"]] \
                = df.loc[:, [label for label in df.columns if label[:5] == "Label"]].fillna(-1)
            df.loc[df_end.index] = df_end
        new_data.append(df)
    new_data = pd.concat(new_data, sort=False)
    new_data.sort_index(inplace=True)
    return new_data


# volatility measure ---------------------------------------------------------------------------------------------------
# get daily vola, equally weighted
def get_vola(close, window=252, weights="equal_weight"):
    if window is None:
        window = close.shape[0]

    if weights == "ewma":  # exp mov av
        logRets = np.log(close / close.shift(1))
        vola = logRets.ewm(span=window, min_periods=10, adjust=True).std()   # TODO limit window (like get_EMA)
    else:  # equal weight
        logRets = np.log(close / close.shift(1))
        vola = logRets.rolling(window=window, min_periods=10).std()  # for annual: * np.sqrt(252)
    return vola


# get Average True Range
def get_ATR(df, window=21):
    true_high = pd.concat([df["High"], df["Close"].shift(1)], axis=1).max(axis=1)
    true_low = pd.concat([df["Low"], df["Close"].shift(1)], axis=1).min(axis=1)
    diff = true_high - true_low
    atr = diff.rolling(window=window, min_periods=1).mean()
    return atr


# get drawdown
def get_DD(close):
    high_watermark = close[0]
    dd = np.zeros(close.shape[0])
    for i in range(1, close.shape[0]):
        high_watermark = max(high_watermark, close[i])
    raw_dd = high_watermark - close[i]
    dd[i] = raw_dd / high_watermark
    return dd


# get rolling drawdown
def get_roll_DD(close, window=None):
    if window is None:
        window = close.shape[0]  # max window
    dd = close.rolling(window=window, min_periods=1).apply(lambda x: (x.max() - x[-1]) / x.max(), raw=True)
    return dd


# trend indicator ------------------------------------------------------------------------------------------------------
# return relative trend (1 iff close == upper envelope, -1 iff close == lower envelope).
# envelope consists of the signal (close), lower and upper envelope
def get_relative_trend(envelope):
    trend = 2 * (envelope.iloc[:, 0] - envelope.Lower) / (envelope.Upper - envelope.Lower) - 1
    return trend.replace(np.inf, trend[np.isfinite(trend)].max()).replace(-np.inf, trend[np.isfinite(trend)].min())


# transform underlying to 0 or 1. envelope consists of the signal (close), lower and upper envelope
def get_discrete_trend(envelope):
    close = envelope.iloc[:, 0]
    hit_low = (close <= envelope.Lower)
    hit_high = (close >= envelope.Upper)

    # first non nan idx
    first_valid_idx = np.max(envelope.reset_index().iloc[:, 1:].notna().idxmax())

    trend = pd.Series(np.zeros(len(close)), index=close.index)
    trend[:first_valid_idx] = np.nan

    # first trend 1 iff close is closer to upper envelope
    trend[first_valid_idx] = 1 * (close[first_valid_idx] > (envelope.Lower[first_valid_idx] + envelope.Upper[first_valid_idx]) / 2)
    for i in range(first_valid_idx + 1, len(close)):
        if hit_low[i] or hit_high[i]:  # only one can be true
            trend[i] = 1 * hit_high[i]   # 1 if high, else 0
        else:
            trend.iloc[i] = trend.iloc[i - 1]
    return trend


# return altered envelope. Apply rolling min or max
def get_adaptive_envelope(envelope, window=250, lower_func="none", upper_func="none", prio_lower_func=True):
    # if window is None:
    #     window = envelope.shape[0]

    adapt_env = envelope.copy(deep=True)

    if lower_func == "min":
        adapt_env.Lower = envelope.Lower.rolling(window=window, min_periods=int(window / 10)).min()
    elif lower_func == "max":
        adapt_env.Lower = envelope.Lower.rolling(window=window, min_periods=int(window / 10)).max()

    if upper_func == "min":
        adapt_env.Upper = envelope.Upper.rolling(window=window, min_periods=int(window / 10)).min()
    elif upper_func == "max":
        adapt_env.Upper = envelope.Upper.rolling(window=window, min_periods=int(window / 10)).max()

    # an envelope requires both upper and lower envelope
    first_valid_idx = np.max(adapt_env[["Lower", "Upper"]].notna().idxmax())
    adapt_env.loc[:first_valid_idx, ["Lower", "Upper"]] = np.NaN

    # lower and upper envelope must not cross
    if prio_lower_func:
        adapt_env.Upper = adapt_env[["Lower", "Upper"]].max(axis=1)
    else:
        adapt_env.Lower = adapt_env[["Lower", "Upper"]].min(axis=1)

    return adapt_env


# market features ------------------------------------------------------------------------------------------------------

# helper func, return series containing the idx slice of a test day
def slice_days(df):
    test_days = df.index.unique()
    curr_day = test_days[0]
    curr_day_pointer = 0
    test_slices = list()
    curr_start_idx = 0
    for i in range(df.shape[0]):
        if df.index[i] > curr_day:  # found first sample of next day
            test_slices.append(slice(curr_start_idx, i))  # save slice
            curr_start_idx = i
            curr_day_pointer += 1
            curr_day = test_days[curr_day_pointer]
    test_slices.append(slice(curr_start_idx, df.shape[0]))
    return pd.Series(test_slices, index=test_days)


# return proportion of SP500 assets that have a positive trend.
# Underlying trend in the form of "Underlying" has to be in dataset
def mrkt_trend(df, underlying=["BollingerTrend250"], remove_trend_suffix=False):
    trade_days = slice_days(df)
    trend = np.zeros(df.shape[0] * len(underlying)).reshape(df.shape[0], -1)
    for curr_day_idx in range(0, trade_days.shape[0]):
        curr_df = df.iloc[trade_days[curr_day_idx]][["IndexMember"] + underlying]
        curr_df = curr_df[curr_df.IndexMember >= 1]  # only index for mean
        up_proportion = curr_df[underlying].mean()  # okay if trend is either 0 or 1
        # up_proportion = curr_df[underlying].apply(lambda x: (x + 1) / 2).mean()  # if -1 or 1
        trend[trade_days[curr_day_idx]] = up_proportion
    if remove_trend_suffix:
        name = [''.join([char for char in word if not char.isdigit()]) for word in underlying]
        number = [''.join([char for char in word if char.isdigit()]) for word in underlying]
        new_features = ["Mrkt" + text[:-5] + nb for text, nb in zip(name, number)]
    else:
        new_features = ["Mrkt" + feature for feature in underlying]

    return pd.DataFrame(trend, index=df.index, columns=new_features)


# get index membership ---------------------------------------------------------------------------------------------
# data should start 1990 (only SP500) or 2000 (SP100 and SP500)
def get_index_membership(df, symbol, idx_member, idx_member_anytime, idx_member_top=None):
    if df.index[0] < pd.to_datetime("1990-01-01"):  # TODO only for SP500
        raise Exception("Data cannot preceed 1990 due to SP500 data restrictions. Asset %s." % symbol)
    if idx_member_top is not None:
        if df.index[0] < pd.to_datetime("2000-01-01"):
            raise Exception("Data cannot preceed 2000 due to SP100 data restrictions. Asset %s." % symbol)
    is_member = np.zeros(df.shape[0], dtype=np.int8)
    if idx_member_top is not None:  # two indexes exist (SP500 and SP100)
        if symbol in idx_member_anytime.values:  # asset has been in index at some point
            curr_idx_member = idx_member.iloc[0, :]  # SP500
            curr_idx_member_top = idx_member_top.iloc[0, :]  # SP100
            curr_idx_member_counter = 0  # to loop through idx_member
            for t in range(0, df.shape[0]):  # loop through days to determine index membership
                while df.index[t] > curr_idx_member.name and curr_idx_member_counter < idx_member.shape[0] - 1:
                    curr_idx_member_counter += 1  # find first match
                    curr_idx_member = idx_member.iloc[curr_idx_member_counter, :]
                    curr_idx_member_top = idx_member_top.iloc[curr_idx_member_counter, :]
                if df.index[t] <= curr_idx_member.name:
                    if symbol in curr_idx_member_top.values:
                        is_member[t] = 2
                    elif symbol in curr_idx_member.values:
                        is_member[t] = 1
                elif curr_idx_member_counter < idx_member.shape[0] - 1:
                    curr_idx_member_counter += 1
                    curr_idx_member = idx_member.iloc[curr_idx_member_counter, :]
                    curr_idx_member_top = idx_member_top.iloc[curr_idx_member_counter, :]
                    if symbol in curr_idx_member_top.values:
                        is_member[t] = 2
                    elif symbol in curr_idx_member.values:
                        is_member[t] = 1
                else:  # end of idx_member
                    if symbol in curr_idx_member_top.values:
                        is_member[t] = 2
                    elif symbol in curr_idx_member.values:
                        is_member[t] = 1
    else:  # only one index (SP500/STOXX)
        if symbol in idx_member_anytime.values:  # asset has been in index at some point
            curr_idx_member = idx_member.iloc[0, :]  # index
            curr_idx_member_counter = 0  # to loop through idx_member
            for t in range(0, df.shape[0]):  # loop through days to determine index membership
                while df.index[t] > curr_idx_member.name and curr_idx_member_counter < idx_member.shape[0] - 1:
                    curr_idx_member_counter += 1  # find first match
                    curr_idx_member = idx_member.iloc[curr_idx_member_counter, :]
                if df.index[t] <= curr_idx_member.name:
                    if symbol in curr_idx_member.values:
                        is_member[t] = 1
                elif curr_idx_member_counter < idx_member.shape[0] - 1:
                    curr_idx_member_counter += 1
                    curr_idx_member = idx_member.iloc[curr_idx_member_counter, :]
                    if symbol in curr_idx_member.values:
                        is_member[t] = 1
                else:  # end of idx_member
                    if symbol in curr_idx_member.values:
                        is_member[t] = 1
    return is_member
