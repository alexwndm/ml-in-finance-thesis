# Backtesting

import numpy as np
import pandas as pd
import pickle

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

from classes.AssetAllocator import EWAllocator
from classes.Filter import NoFilter

from functions.feature_engineering import get_roll_DD, slice_days

matplotlib.use('Agg')

# Walk-forward backtester
class WFBacktester:
    def __init__(self, filter=None, allocator=None, realloc_period=1, start_equity=1000,
                 output_path="./output/Backtests/"):
        # asset filter
        if filter is None:
            self.filter = NoFilter()
        else:
            self.filter = filter

        # asset allocation
        if allocator is None:
            self.allocator = EWAllocator()  # equal weight
        else:
            self.allocator = allocator
        self.realloc_period = realloc_period

        # set start equity
        self.start_equity = start_equity
        # output path
        self.output_path = output_path

    # methods ----------------------------------------------------------------------------------------------------------
    # portfolio initialization
    def init_portfolio(self, curr_df):
        portfolio_symbols = self.filter.choose_assets(curr_df)
        portfolio_weights = self.allocator.get_weights(curr_df, portfolio_symbols)
        portfolio_equity = portfolio_weights * self.start_equity
        date = pd.DatetimeIndex(np.repeat(self.test_days_.index[0], portfolio_weights.shape[0]))
        portfolio = pd.DataFrame({"AssetSymbol": np.append("Cash", portfolio_symbols),
                                  "Weight": portfolio_weights,
                                  "Equity": portfolio_equity},
                                 index=date)
        return portfolio

    # run a backtest
    def run(self, df):
        print("Start Backtest on %s." % str(df.index[0])[:10])
        # get test days
        self.test_days_ = slice_days(df)

        # get time dependent asset universe
        asset_universe = list()
        # save number of index members (lower than asset_universe, which contains assets not yet/anymore in index)
        index_member_nb = 0
        for curr_day_idx in range(0, self.test_days_.shape[0]):
            curr_df = df.iloc[self.test_days_[curr_day_idx]]
            asset_universe.append(curr_df["AssetSymbol"].unique())
            index_member_nb = max(index_member_nb, curr_df.loc[curr_df.IndexMember >= 1, "AssetSymbol"].shape[0])
        self.asset_universe_ = pd.Series(asset_universe, index=self.test_days_.index, name=index_member_nb)

        # init portfolio
        self.equity_ = pd.DataFrame()
        self.equity_.loc[self.test_days_.index[0], "Equity"] = self.start_equity
        self.portfolio_ = self.init_portfolio(curr_df=df.iloc[self.test_days_[0]])  # (Cash, Assets)
        curr_portfolio = self.portfolio_.copy(deep=True)
        bankruptcy_list = list()
        self.portfolio_size_ = pd.DataFrame()
        self.portfolio_size_.loc[self.test_days_.index[0], "PortfolioSize"] = curr_portfolio.shape[0] - 1

        realloc_day_idx = self.realloc_period
        for curr_day_idx in range(1, self.test_days_.shape[0]):  # loop through days
            curr_df = df.iloc[self.test_days_[curr_day_idx]]
            for asset in curr_portfolio["AssetSymbol"][1:]:  # loop through assets
                if asset in self.asset_universe_.iloc[curr_day_idx]:  # disappearing assets stay flat
                    ret = curr_df.loc[curr_df["AssetSymbol"] == asset, "Return1"][0]
                    curr_portfolio.loc[curr_portfolio["AssetSymbol"] == asset, "Equity"] *= (1 + ret)
                # else:
                #     curr_portfolio.loc[curr_portfolio["AssetSymbol"] == asset, "Equity"] = 0  # bankruptcy -> only penny stocks < 5
            # update weights
            curr_equity = curr_portfolio["Equity"].sum()
            curr_portfolio.Weight = curr_portfolio["Equity"] / curr_equity
            curr_portfolio.index = pd.DatetimeIndex(np.repeat(self.test_days_.index[curr_day_idx],
                                                              curr_portfolio.shape[0]))

            if curr_day_idx >= realloc_day_idx:  # reallocation
                if curr_portfolio[1:].loc[curr_portfolio[1:]["Weight"] == 0].shape[0] > 0:
                    bankruptcy_list.append(
                        pd.DataFrame(curr_portfolio[1:].loc[curr_portfolio[1:]["Weight"] == 0, "AssetSymbol"]))
                print("Reallocation on %s." % str(self.test_days_.index[curr_day_idx])[:10])
                curr_equity = curr_portfolio["Equity"].sum()  # liquidate everything
                portfolio_symbols = self.filter.choose_assets(curr_df)  # filter assets
                portfolio_weights = self.allocator.get_weights(curr_df, portfolio_symbols)  # asset allocation
                portfolio_equity = portfolio_weights * curr_equity  # allocation
                # create new portfolio
                curr_portfolio = pd.DataFrame({"AssetSymbol": np.append("Cash", portfolio_symbols),
                                               "Weight": portfolio_weights,
                                               "Equity": portfolio_equity},
                                              index=[np.repeat(self.test_days_.index[curr_day_idx],
                                                               portfolio_weights.shape[0])])
                # update portfolio size
                self.portfolio_size_.loc[self.test_days_.index[curr_day_idx],
                                         "PortfolioSize"] = curr_portfolio.shape[0] - 1
                # update next reallocation day
                realloc_day_idx = curr_day_idx + self.realloc_period

            # save portfolio
            self.portfolio_= self.portfolio_.append(curr_portfolio)  # update portfolio history
            self.equity_.loc[self.test_days_.index[curr_day_idx]] = curr_equity  # update equity

        if len(bankruptcy_list) == 0:
            self.bankrupt_assets_ = None
        else:
            self.bankrupt_assets_ = pd.concat(bankruptcy_list)
            print("Bancruptcy: \n", self.bankrupt_assets_)

        # self.save_backtest()
        self.summary()

    # export backtest statistics
    def summary(self):
        # plot equity
        plot_data(self.equity_, self.output_path)
        plot_data(self.portfolio_size_, self.output_path, ylim=self.asset_universe_.name)
        # plot portfolio weights
        # plot_weights(self.portfolio_)
        # stats
        time_period = self.test_days_.index[0].strftime("%Y-%m-%d") + " to " \
                      + self.test_days_.index[-1].strftime("%Y-%m-%d")
        if self.filter.method is not "MLFilter":
            method = self.filter.method
        else:
            method = "MLFilter with " + str([clf.model_class + clf.label_ for clf in self.filter.classifier_list])
        portfolio_size = (self.portfolio_.Weight > 0).sum() / self.test_days_.shape[0]
        total_return = self.equity_.iloc[-1] / self.start_equity - 1
        geom_return = ((total_return + 1) ** (252.0 / self.test_days_.shape[0]) - 1)[0]
        vola = (np.std(self.equity_ / self.equity_.shift(1) - 1) * np.sqrt(252))[0]
        # vola = get_vola(self.equity_, window=None, weights="equal_weight").iloc[-1] * np.sqrt(252)
        drawdown = get_roll_DD(self.equity_, window=None)
        drawdown.columns = ["Drawdown"]
        plot_data(drawdown, self.output_path)

        # time_under_water = (drawdown > 0).sum() / drawdown.shape[0]
        sharpe_ratio = geom_return / vola

        summary = "Backtest summary \n" \
                  "-------------------------------------\n" \
                  "time period: %s\n" \
                  "filter: %s\n" \
                  "average portfolio size: %.4f\n" \
                  "asset allocator: %s\n" \
                  "reallocation period: %d days\n" \
                  "start equity: %d\n" \
                  "end equity: %d\n" \
                  "total return: %.4f%%\n" \
                  "geom. yearly return: %.4f%%\n" \
                  "yearly vola: %.4f\n" \
                  "DD (arithm. mean): %.4f%%\n" \
                  "DD (geom. mean): %.4f%%\n" \
                  "DD (max): %.4f%%\n" \
                  "sharpe ratio: %.4f\n" \
                  % (time_period,
                     method,
                     portfolio_size,
                     self.allocator.method,
                     self.realloc_period,
                     self.start_equity,
                     self.equity_.values[-1][0],
                     total_return * 100,
                     geom_return.mean() * 100,
                     vola,
                     drawdown.mean() * 100,
                     (np.exp(np.mean(np.log(drawdown + 1))) - 1)[0] * 100,
                     drawdown.max() * 100,
                     sharpe_ratio)
        # export summary
        stats_file = open(self.output_path + "backtest_summary.txt", "w")
        stats_file.write(summary)
        stats_file.close()
        print(summary)

    def save_backtest(self, file="CurrBacktest"):
        filepath = self.output_path + file
        with open(filepath, 'wb') as curr_file:
            pickle.dump(self, curr_file)


# functions ------------------------------------------------------------------------------------------------------------
# load backtest
def load_backtest(file="CurrBacktest"):
    filepath = "./output/Backtests/" + file
    with open(filepath, 'rb') as curr_file:
        model = pickle.load(curr_file)
    return model


# plots ----------------------------------------------------------------------------------------------------------------
# plot data (portfolio equity, drawdown, portflio_size...)
def plot_data(data, output_path, ylim=None):
    data = data.reset_index()
    data.columns = ["Date", data.columns[1]]
    register_matplotlib_converters()
    sns.set(style="whitegrid", font_scale=2)
    plt.grid(which="minor")
    fig, ax = plt.subplots(figsize=(16, 8))
    if ylim is not None:
        plt.ylim(0, ylim)
    sns.lineplot(x="Date", y=data.columns[1], data=data, ax=ax)

    # ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    plt.savefig(output_path + data.columns[1] + ".png")
    plt.close("all")


# # plot portfolio weights  # TODO plot weights...
# def plot_weights(portfolio):
#     # register_matplotlib_converters()
#     fig1, ax1 = plt.subplots()
#     ax1.pie(portfolio, explode=explode, labels=labels, autopct='%1.1f%%',
#             shadow=True, startangle=90)
#     ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
#     # plt.show()
#     plt.show()
#




