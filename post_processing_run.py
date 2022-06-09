# Applies Post Processing.
# Uses the equity curves that have been generated with backtest_run.py.
# Applies a number of trend indicators to the equity curves.
# Output in the subfile output/Backtests/PostProcessing.

import pandas as pd
import numpy as np

from functions.feature_engineering import get_roll_DD, get_discrete_trend
from functions.plots import plot_equity
from functions.MrktIndicators.Envelope import get_envelope
from functions.MrktIndicators.Bollinger import get_bollinger_band
from functions.MrktIndicators.Donchian import get_donchian_channel
from functions.MrktIndicators.MACD import get_MACD_envelope
from functions.MrktIndicators.Oscillator import get_osci_envelope


def run():
    equity_df_SP = pd.read_csv("./output/Backtests/PostProcessing/_SP/Equity_df_SP.csv", sep=";",
                               index_col=0, parse_dates=True)
    equity_df_STOXX = pd.read_csv("./output/Backtests/PostProcessing/_STOXX/Equity_df_STOXX.csv", sep=";",
                                  index_col=0, parse_dates=True)
    print("Apply trend indicators to equity curves.")
    apply_trend_indicators(equity_df_SP, index="SP")
    apply_trend_indicators(equity_df_STOXX, index="STOXX")
    print("Apply post processing on S&P 500 dataset.")
    apply_post_processing(equity_df_SP, trend_df=None, index="SP")
    apply_summary(equity_df_SP.columns, index="SP")

    print("Apply post processing on STOXX Europw 600 dataset.")
    apply_post_processing(equity_df_STOXX, trend_df=None, index="STOXX")
    apply_summary(equity_df_STOXX.columns, index="STOXX")


def apply_trend_indicators(equity_df, index="SP"):
    print("Apply trend to index.")
    if index == "SP":
        index_df = pd.read_csv("./data/SP500TR.csv", index_col=0, parse_dates=True)
        index_df = index_df[index_df.index >= pd.to_datetime("2001-01-01")]
        index_df = index_df[index_df.index <= pd.to_datetime("2008-01-01")]
    elif index == "STOXX":
        index_df = pd.read_csv("./data/STOXX600TR.csv", index_col=0, parse_dates=True)
        index_df = index_df[index_df.index >= pd.to_datetime("2007-01-01")]
        index_df = index_df[index_df.index <= pd.to_datetime("2013-01-01")]
    else:
        print("Index dataset not found.")

    trend_df_idx = pd.DataFrame(index=index_df.index)
    for period in range(100, 201, 10):
        # add envelope
        envelope = get_envelope(index_df, window=period, min_periods=10, envelope_prct=0.03)
        trend_df_idx["EnvelopeTrend" + str(period) + "(Days)"] = get_discrete_trend(envelope)
        # add Bollinger Band
        boll = get_bollinger_band(index_df, window=period, min_periods=10, std_factor=2)
        trend_df_idx["BollingerTrend" + str(period) + "(Days)"] = get_discrete_trend(boll)
        # add Donchian channel
        donchian = get_donchian_channel(index_df, window=period)
        trend_df_idx["DonchianTrend" + str(period) + "(Days)"] = get_discrete_trend(donchian)

    for factor in range(10, 21, 1):
        # add MACD
        macd = get_MACD_envelope(index_df, window=None, streched=False, factor=factor)
        trend_df_idx["MACDTrend" + str(factor) + "(Factor)"] = get_discrete_trend(macd)
    for alpha in range(4, 15):
        # add gradient oscillator
        # osci = get_osci_envelope(equity, alpha=alpha, window=100, time_restriction=False)
        osci = get_osci_envelope(index_df, alpha=alpha, window=100, time_restriction=True)
        trend_df_idx["OscillatorTrend" + str(alpha) + "(Alpha)"] = get_discrete_trend(osci)

    for j in range(equity_df.shape[1]):  # for every trading strategy
        equity = equity_df.iloc[:, j]
        print("Equity curve of %s." % equity.name)

        trend_df = pd.DataFrame(index=equity_df.index)
        for period in range(100, 201, 10):
            # add envelope
            envelope = get_envelope(equity, window=period, min_periods=10, envelope_prct=0.03)
            trend_df["EnvelopeTrend" + str(period) + "(Days)"] = get_discrete_trend(envelope)
            # add Bollinger Band
            boll = get_bollinger_band(equity, window=period, min_periods=10, std_factor=2)
            trend_df["BollingerTrend" + str(period) + "(Days)"] = get_discrete_trend(boll)
            # add Donchian channel
            donchian = get_donchian_channel(equity, window=period)
            trend_df["DonchianTrend" + str(period) + "(Days)"] = get_discrete_trend(donchian)

        for factor in range(10, 21, 1):
            # add MACD
            macd = get_MACD_envelope(equity, window=None, streched=False, factor=factor)
            trend_df["MACDTrend" + str(factor) + "(Factor)"] = get_discrete_trend(macd)
        for alpha in range(4, 15):
            # add gradient oscillator
            # osci = get_osci_envelope(equity, alpha=alpha, window=100, time_restriction=False)
            osci = get_osci_envelope(equity, alpha=alpha, window=100, time_restriction=True)
            trend_df["OscillatorTrend" + str(alpha) + "(Alpha)"] = get_discrete_trend(osci)

        # initialize trend with index trend
        trend_df[:250] = trend_df_idx[trend_df_idx.index >= trend_df.index[0]][:250]
        # export
        trend_df.to_csv("./output/Backtests/PostProcessing/_" + index + "/" + equity.name + "/Trend_df.csv", sep=";")
    return


# linear (old)
def post_processing(equity, trend):
    performance = equity / equity.shift(1)  # daily performance
    new_equity = equity.copy(deep=True)
    trend = trend.shift(1)  # trend based on close
    trend = trend[trend.index >= equity.index[0]]
    for i in range(1, equity.shape[0]):
        if trend.iloc[i]:  # good market trend
            new_equity.iloc[i] = new_equity.iloc[i-1] * performance.iloc[i]  # invested, performance of equity curve
        else:  # bad market trend
            new_equity.iloc[i] = new_equity.iloc[i - 1]  # go flat
    return new_equity


# vectorized
def apply_post_processing(equity_df, trend_df, index="SP"):
    if trend_df is not None:  # one index trend df for all
        trend_df = trend_df.shift(1)  # trend based on close
        trend_df = trend_df[trend_df.index >= equity_df.index[0]]  # sync trend and equity df
    for j in range(equity_df.shape[1]):  # for every trading strategy
        equity = equity_df.iloc[:, j]
        print("Post processing for equity curve of %s." % equity.name)
        if trend_df is None:
            trend_df = pd.read_csv("./output/Backtests/PostProcessing/_" + index + "/" + equity.name + "/Trend_df.csv",
                                   sep=";", index_col=0, parse_dates=True)
            trend_df = trend_df.shift(1)  # trend based on close
            trend_df = trend_df[trend_df.index >= equity_df.index[0]]  # sync trend and equity df
        performance = equity / equity.shift(1)  # daily performance (1 + ret)
        new_col = [equity.name + " & " + col for col in trend_df.columns]
        equity_df_out = pd.DataFrame(index=equity_df.index, columns=new_col)
        equity_df_out.iloc[0, :] = equity.values[0]  # set start equity
        for i in range(1, equity.shape[0]):
            factor = trend_df.iloc[i, :].replace(1.0, performance[i]).\
                replace([0.0, np.nan], 1).values  # go flat if trend == 0
            equity_df_out.iloc[i, :] = equity_df_out.iloc[i - 1, :].multiply(factor)
        strategies = list(pd.Series([col.split("Trend")[0] + "Trend" for col in trend_df.columns]).unique())
        strat_col = [equity.name + " & " + s for s in strategies]
        equity_df_mean = pd.DataFrame(index=equity_df.index, columns=strat_col)
        for strat in strat_col:
            equity_df_mean.loc[:, strat] = equity_df_out.loc[:,
                                           [x for x in equity_df_out.columns
                                            if x.split("Trend")[0] + "Trend" == strat]].mean(axis=1)
        equity_df_out = pd.concat([equity, equity_df_out, equity_df_mean], axis=1)
        equity_df_out.to_csv("./output/Backtests/PostProcessing/_" + index + "/" +
                             equity.name + "/Equity_df_post_processed.csv", sep=";")
    return


# test statistics
def summarize_equity(equity_df, index="SP"):
    statistics = ["Time Period", "Start Equity", "End Equity", "Total Return", "Annualized Return", "Annualized Vola",
                  "Average Drawdown", "Maximum Drawdown", "Sharpe Ratio (Vola)", "Sharpe Ratio (DD)"]
    summary_df = pd.DataFrame(index=statistics, columns=equity_df.columns)
    for j in range(equity_df.shape[1]):  # for every strategy
        equity = equity_df.iloc[:, j]
        summary_df.loc["Time Period", equity.name] = equity.index[0].strftime("%Y-%m-%d") + " to " \
                                                     + equity.index[-1].strftime("%Y-%m-%d")
        start_equity = equity.iloc[0]
        summary_df.loc["Start Equity", equity.name] = start_equity
        end_equity = equity.iloc[-1]
        summary_df.loc["End Equity", equity.name] = end_equity
        total_return = end_equity / start_equity - 1
        summary_df.loc["Total Return", equity.name] = total_return
        geom_return = ((total_return + 1) ** (252.0 / equity.shape[0]) - 1)
        summary_df.loc["Annualized Return", equity.name] = geom_return
        vola = (np.std(equity / equity.shift(1) - 1) * np.sqrt(252))
        summary_df.loc["Annualized Vola", equity.name] = vola
        drawdown = get_roll_DD(equity, window=None)
        av_dd = drawdown.mean()
        summary_df.loc["Average Drawdown", equity.name] = av_dd
        summary_df.loc["Maximum Drawdown", equity.name] = drawdown.max()
        summary_df.loc["Sharpe Ratio (Vola)", equity.name] = geom_return / vola
        summary_df.loc["Sharpe Ratio (DD)", equity.name] = geom_return / av_dd

    summary_df.to_csv("./output/Backtests/PostProcessing/_" + index + "/" +
                      equity_df.iloc[:, 0].name + "/Equity_df_pp_summary.csv",
                      sep=";")
    return


# strategies = ['No Filter & EW', 'No Filter & HRP', 'Vola Basket & EW', 'Vola Basket & HRP',
#               'ML Selection, p ≥ 0.5 & EW', 'ML Selection, p ≥ 0.5 & HRP']
def apply_summary(strategies, index="SP"):
    summary_list = list()
    for strat in strategies:
        print("Summarize %s." % strat)
        post_df = pd.read_csv("./output/Backtests/PostProcessing/_" + index + "/" + strat +
                              "/Equity_df_post_processed.csv",
                              sep=";", index_col=0, parse_dates=True)
        plot_equity(post_df.iloc[:, [0, -5, -4, -3, -2, -1]], "/PostProcessing/_" + index + "/" + strat + "/Equity_Plot")
        summarize_equity(post_df, index=index)
        sum_df = pd.read_csv("./output/Backtests/PostProcessing/_" + index + "/" + strat +
                             "/Equity_df_pp_summary.csv",
                             sep=";", index_col=0, parse_dates=True)
        cols = [col.split(" & ")[-1] for col in sum_df.columns]
        cols[0] = "Original"
        sum_df.columns = cols
        statistics = ["Start Equity", "End Equity", "Total Return", "Annualized Return",
                      "Annualized Vola", "Average Drawdown", "Maximum Drawdown",
                      "Sharpe Ratio (Vola)", "Sharpe Ratio (DD)"]
        summary_list.append(sum_df.loc[statistics].astype(float))
    sum_df_mean = summary_list[0].add(summary_list[1]). \
                      add(summary_list[2]).add(summary_list[3]).add(summary_list[4]).add(summary_list[5]) / 6
    sum_df_mean.to_csv("./output/Backtests/PostProcessing/_" + index + "/" + "Summary.csv", sep=";")
    Ret = sum_df_mean.loc["Annualized Return"].sort_values(ascending=False)
    Ret.to_csv("./output/Backtests/PostProcessing/_" + index + "/" + "Best_Return.csv", sep=";")
    SR = sum_df_mean.loc["Sharpe Ratio (Vola)"].sort_values(ascending=False)
    SR.to_csv("./output/Backtests/PostProcessing/_" + index + "/" + "Best_SR.csv", sep=";")
    SR_dd = sum_df_mean.loc["Sharpe Ratio (DD)"].sort_values(ascending=False)
    SR_dd.to_csv("./output/Backtests/PostProcessing/_" + index + "/" + "Best_SR_DD.csv", sep=";")
    return


if __name__ == '__main__':
    run()
