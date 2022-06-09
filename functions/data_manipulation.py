# functions to read and alter data

# general
import os
import pandas as pd
import numpy as np

# other
from functions.feature_engineering import add_features, get_index_membership, mrkt_trend


# read data
def read_data(filepath, index,  nb_of_assets=None, return_horizons=[1, 5, 10, 30, 50], label_horizons=10,
              value="CloseAdj", label_type="triple_barrier", vola_weights="equal_weights"):
    cwd = os.getcwd()  # get current directory
    os.chdir(filepath)
    filenames = [x for x in os.listdir() if
                 x.endswith('.csv') and os.path.getsize(x) >= 6000]  # TODO adjust size, 7KB~400Trading Days
    np.random.seed(0)
    if nb_of_assets is not None:
        filenames = np.random.choice(filenames, nb_of_assets, replace=False)
    print("Using data of %d assets." % (len(filenames)))

    # index
    print(os.getcwd())
    idx_member = pd.read_csv("../../IndexMember/" + index + "_TR.CSV",
                             index_col=0, parse_dates=True, header=None, dtype=object)
    idx_member_anytime = pd.read_csv("../../IndexMember/" + index + "_TR_anytime.CSV",
                                     header=None)
    idx_member = idx_member.shift(1).iloc[1:]  # index represents membership end date

    print("Checking index membership.")
    data = list()
    for filename in filenames:
        df = pd.read_csv(filename, index_col="Date", parse_dates=True)
        symbol1, symbol2, _ = filename.split(sep=" ")
        symbol = symbol1 + "_" + symbol2
        df["AssetSymbol"] = symbol
        # get index membership
        df["IndexMember"] = get_index_membership(df=df, symbol=symbol,
                                                 idx_member=idx_member, idx_member_anytime=idx_member_anytime)
                                                 # idx_member_top=idx_member_top)
        data.append(df)
    os.chdir(cwd)

    data = add_features(data, return_horizons=return_horizons, label_horizons=label_horizons,
                        value=value, label_type=label_type, vola_weights=vola_weights)
    # add market data
    print("Calculating market trend.")
    for h in return_horizons:
        print("Calculating with a window of %s days." % h)
        # normal features
        underlying = ["Return", "Vola", "DD"]
        underlying = [feature + str(h) for feature in underlying]
        mrkt_data = mrkt_trend(data, underlying)  # calculate market features
        data = pd.concat([data, mrkt_data], axis=1)

        # # mrkt indicators
        # underlying = ["EnvelopeTrend", "BollingerTrend", "DonchianTrend",
        #               "MACDTrend", "OscillatorTrend"]
        # underlying = [feature + str(h) for feature in underlying]
        # mrkt_data = mrkt_trend(data, underlying, remove_trend_suffix=True)  # calculate market features
        # data = pd.concat([data, mrkt_data], axis=1)
        # for f1, F1 in zip(["none", "min", "max"], ["None", "Min", "Max"]):
        for f1, F1 in zip(["min"], ["Min"]):
            for f2, F2 in zip(["max"], ["Max"]):
                underlying = ["Env", "Boll", "MACD", "Osci"]
                underlying = [feature + F1 + F2 + "Trend" + str(h) for feature in underlying]
                mrkt_data = mrkt_trend(data, underlying, remove_trend_suffix=True)  # calculate market features
                data = pd.concat([data, mrkt_data], axis=1)
    return data


# add market data
def read_index_data(filepath, vix_filepath=None, return_horizons=[1, 10, 50], rolling_windows=[200],
                    value="CloseAdj", label_type="triple_barrier", vola_weights="equal_weights"):
    # df = pd.read_csv(filepath, sep=",")
    # df["Date"] = pd.to_datetime(df["Date"])
    # df = df.set_index("Date")
    # df.sort_index(inplace=True)
    df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
    df = add_features([df], return_horizons=return_horizons, rolling_windows=rolling_windows, label_horizons=None,
                      value=value, label_type=label_type, vola_weights=vola_weights, is_index=True)
    df.columns = "Index" + df.columns
    if vix_filepath is not None:
        vix_df = pd.read_csv(vix_filepath, sep=",")
        vix_df["Date"] = pd.to_datetime(vix_df["Date"])
        vix_df = vix_df.set_index("Date")
        vix_df.sort_index(inplace=True)
        vix_df["VIX"] = vix_df["VIX Close"]
        df = pd.concat([df, vix_df["VIX"]], join="inner", axis=1)
    return df


# set start date
def set_start_date(df, year=1900, month=1, day=1):
    start_date = pd.Timestamp(year, month, day)
    return df[df.index > start_date]


# functions to clean dataset -------------------------------------------------------------------------------------------

# delete duplicates in dataset if asset vanished at some point (then the last value is repeated endlessly)
def del_dupl(df):
    close_values = df["INDX_MEMBERS"].value_counts()
    most_common_nb = close_values.values[0]
    most_common_value = close_values.index[0]
    most_common_is_last = (most_common_value == df["INDX_MEMBERS"].values[-1])
    if most_common_nb >= 3 and most_common_is_last:
        dupl_dates = df[df["INDX_MEMBERS"] == most_common_value].index
        last_dates = df[df.index >= dupl_dates[0]].index
        # don't delete duplicates too early
        dupl_in_last_dates = pd.Series(last_dates.isin(dupl_dates), index=last_dates)
        if (~dupl_in_last_dates).sum() > 0:  # if one of duplicates appears early on (randomly): keep it
            drop_dates = df[df.index > dupl_in_last_dates[~dupl_in_last_dates].index[-1]].index[1:]
        else:
            drop_dates = dupl_dates[1:]
        df = df.drop(drop_dates)  # drop duplicates
    return df


# main cleaning function
def clean_dataset():
    os.chdir("./data/Marktdaten/STOXX600TR/Equities/")
    new_filepath = "./data/Marktdaten/STOXX600TR/Equities_clean_new/"
    filenames = [x for x in os.listdir() if x.endswith('.csv') and os.path.getsize(x) >= 6000]
    counter = 1
    for filename in filenames:
        print("Cleaning file %s (%d of %d)." % (filename, counter, len(filenames)))
        counter += 1
        df = pd.read_csv(filename, sep=";", index_col="TIME", parse_dates=True, dayfirst=True)
        df = df[df.index >= pd.to_datetime("2002-01-01")]
        df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: float(x.replace(",", ".")))
        df = del_dupl(df)
        df.columns = ["CloseAdj"]
        df.index.name = "Date"
        df.to_csv(new_filepath + filename)
    print("Done.")
    return None


def clean_index_member_dataset(index="STOXX600"):
    # save IndexMember as csv, first line [0,1] etc.
    df = pd.read_csv("./data/Marktdaten/IndexMember/STOXX_raw.csv", header=0, sep=";",
                     index_col=0, parse_dates=True, dayfirst=True)
    symbol_list = list()
    for i in range(df.shape[0]):
        symbols = df.values[i][0].split(sep=",")
        symbols = [symbol.split(" ")[0].replace("/", "_") + "_" + symbol.split(" ")[1] for symbol in symbols]
        symbol_list.append(symbols)
    df0 = pd.DataFrame(symbol_list, index=df.index)
    df0.to_csv("./data/Marktdaten/IndexMember/" + index + "_TR.csv", header=False)
    unique_symbols = pd.DataFrame(pd.unique(df0.values.reshape(-1,))).dropna()
    unique_symbols.to_csv("./data/Marktdaten/IndexMember/" + index + "_TR_anytime.csv",
                          header=False, index=False)
    return None


def export_features(df, title="CurrFeatures"):
    output_file = open("./data/" + title + ".txt", "w")
    features = "',\n'".join(df.columns)
    output_file.write("[\n'")
    output_file.writelines(features)
    output_file.write("'\n]")
    output_file.close()
    return None
