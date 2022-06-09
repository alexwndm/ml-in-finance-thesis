# functions for plotting -----------------------------------------------------------------------------------------------

# general
import pandas as pd
# plot
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import MultipleLocator


# plot single asset
def plot_single_asset(df):
    data = [go.Candlestick(x=df.index,
                           open=df['Open'], close=df['Close'],
                           high=df['High'], low=df['Low'])]
    fig = go.Figure(data=data)
    fig.update_layout(title=str(df["AssetSymbol"][0]) + ", daily")
    plot(fig)


# plot data
def plot_data(df):
    if len(df["AssetSymbol"].unique()) > 30:
        print("Too many assets to plot.")
        return None

    # Add market index if necessary
    # mrkt_data = pd.DataFrame({"Close": df["MrktClose"].values,
    #                           "AssetSymbol": np.array(['SP500' for _ in range(df.shape[0])], dtype=object)},
    #                          index=df.index)
    # df = pd.concat([df, mrkt_data], sort=True)

    fig = px.line(df.reset_index(), x="Date", y="Close", color="AssetSymbol")
    fig.show()


def plot_heatmap(df, title="Feature Correlation", show_plot=False):
    fig, ax = plt.subplots()
    ax = sns.heatmap(df.corr(), cbar=True, square=True)
    fig.subplots_adjust(top=0.94, bottom=0.35, left=0.28)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_ylim(len(df.corr()), 0)
    ax.set_title(title)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    fig.tight_layout()
    plt.savefig('./output/FeatureCorrelations/%s.png' % title)
    if show_plot:
        plt.show()
    plt.close()


def plot_mrkt_indicators(envelope, title=None, filepath=None):
    df0 = pd.melt(envelope.reset_index(), id_vars=["Date"])
    df0.columns = ["Date", "Legend", envelope.columns[0]]
    sns.set(style="whitegrid", font_scale=2)
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.grid(which="minor")
    # fig.subplots_adjust(top=0.953, bottom=0.147, left=0.086, right=0.978, hspace=0.2, wspace=0.2)
    # fig.subplots_adjust(top=0.911, bottom=0.147, left=0.086, right=0.978, hspace=0.2, wspace=0.2)
    ax = sns.lineplot(x="Date", y=df0.columns[-1], hue="Legend", data=df0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    if filepath is not None:
        fig.savefig(filepath + title + ".png")
        plt.close()
    else:
        fig.show()


def plot_equity(equity_df, filename="Equity_plot"):
    equity = pd.melt(equity_df.reset_index(), id_vars="Date", var_name="Method", value_name="Equity")

    fig, ax = plt.subplots(figsize=(16, 8))
    plt.grid(which="minor")
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.set_yscale("log")
    # ax.xaxis.set_major_locator(mdates.YearLocator())
    sns.set(style="whitegrid", font_scale=1.75)

    sns.lineplot(x="Date", y="Equity", data=equity, hue="Method")
    plt.tight_layout()
    plt.savefig("./output/Backtests/" + filename + ".png")
    plt.close()
