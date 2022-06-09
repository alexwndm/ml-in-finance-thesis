import pandas as pd
import time

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates

from classes.Backtester import WFBacktester, load_backtest
from classes.RandomForestClassifier import RPRandomForestClassifier
from classes.XGBoostClassifier import RPXGBoostClassifier
from classes.Filter import NoFilter, MLFilter, VolaFilter
from classes.AssetAllocator import EWAllocator, HRPAllocator
from functions.train_test_splits import get_train_test_indices
import features as f

# read data ------------------------------------------------------------------------------------------------------------
start_time = time.time()
print("Read data.")
df = pd.read_csv("./data/TestDatasetSP.csv", index_col="Date", parse_dates=True)
# df = pd.read_csv("./data/TestDatasetSTOXX.csv", index_col="Date", parse_dates=True)

train_indices, test_indices = get_train_test_indices(df.index, test_size=0.5, purge_window=120)

# select features

features10 = [f for f in f.features_10to250 if f[-2:] == "10"]
features25 = [f for f in f.features_10to250 if f[-2:] == "25"]
features50 = [f for f in f.features_10to250 if f[-2:] == "50" and not f[-3:] == "150" and not f[-3:] == "250"]
features100 = [f for f in f.features_10to250 if f[-3:] == "100"]
features150 = [f for f in f.features_10to250 if f[-3:] == "150"]
features200 = [f for f in f.features_10to250 if f[-3:] == "200"]
features250 = [f for f in f.features_10to250 if f[-3:] == "250"]

features1 = f.features_10to250
features2 = f.features_10to250
features3 = f.features_10to250

# specify models for MLFilter
models = list()
feature_list = list()
labels = list()
# Random Forest 1 ----------
best_params = {'n_estimators': 250, 'criterion': 'gini', 'min_weight_fraction_leaf': 0.02,
               'max_features': 1, 'max_leaf_nodes': None, 'bootstrap': True, 'oob_score': True,
               'class_weight': 'balanced_subsample'}
models.append(RPRandomForestClassifier(best_params))
feature_list.append(features1)
labels.append(["Label60"])
# RF 2 ---------------------
best_params = {'n_estimators': 250, 'criterion': 'entropy', 'min_weight_fraction_leaf': 0.03,
               'max_features': 2, 'max_leaf_nodes': None, 'bootstrap': True, 'oob_score': True,
               'class_weight': 'balanced_subsample'}
models.append(RPRandomForestClassifier(best_params))
feature_list.append(features2)
labels.append(["Label90"])
# RF 3 ---------------------
best_params = {'n_estimators': 250, 'criterion': 'gini', 'min_weight_fraction_leaf': 0.02,
               'max_features': 3, 'max_leaf_nodes': None, 'bootstrap': True, 'oob_score': True,
               'class_weight': 'balanced_subsample'}
models.append(RPRandomForestClassifier(best_params))
feature_list.append(features3)
labels.append(["Label120"])
# Boosted Tree -------------
# label = ["Label120"]
# pos_label_count = sum(n > 0 for n in df_test[label].values)[0]  # class weights. For binary classes only
# best_params = {'n_estimators': 50, 'max_depth': 7, 'learning_rate': 0.05, "importance_type": "gain",
#                "scale_pos_weights": (df_test[label].shape[0] - pos_label_count) / pos_label_count}
# models.append(RPXGBoostClassifier(best_params))
# labels.append(label)


# completete return matrix for asset allocator. Access to "future" returns is denied in the backtest process
ret_mat = pd.pivot(index="Date",
                   columns="AssetSymbol",
                   values="Return1",
                   data=df[["AssetSymbol", "Return1"]].reset_index())

# prepare dataset for backtesting. Drop labels, pick test samples, drop NAs (only feature NAs at beginning of asset)
df_test = df.iloc[test_indices].drop([label for label in df.columns if label[:5] == "Label"], axis=1).dropna()
equity = list()

# specify backtest parameters ##########################################################################################
min_close_price = 5
max_weight = 0.02
ret_window = 5  # in years
realloc_period = 60  # in days
# filters = ["ML"]
filters = ["None", "Vola", "ML"]
# allocators = ["EW"]
allocators = ["EW", "HRP"]
max_rel_size = 0.2  # desired max portfolio size relative to asset universe
up_proba = 0.51  # minimal predicted up probability required to buy asset
realloc_nb_til_retrain = 4

# No filter, Equal Weight ----------------------------------------------------------------------------------------------
if "None" in filters:
    if "EW" in allocators:
        BacktesterObj = WFBacktester(filter=NoFilter(min_close_price=min_close_price),
                                     allocator=EWAllocator(max_weight=max_weight),
                                     realloc_period=realloc_period, output_path="./output/Backtests/EW/")
        BacktesterObj.run(df_test)
        equity.append(pd.DataFrame(BacktesterObj.equity_.values,
                      index=BacktesterObj.equity_.index,
                      columns=["NoFilter + EW"]))

    if "HRP" in allocators:
        # No filter, HRP
        BacktesterObj = WFBacktester(filter=NoFilter(min_close_price=min_close_price),
                                     allocator=HRPAllocator(ret_mat=ret_mat, max_weight=max_weight, ret_window=ret_window),
                                     realloc_period=realloc_period, output_path="./output/Backtests/HRP/")
        BacktesterObj.run(df_test)
        equity.append(pd.DataFrame(BacktesterObj.equity_.values,
                      index=BacktesterObj.equity_.index,
                      columns=["NoFilter + HRP"]))

# Vola filter, Equal Weight --------------------------------------------------------------------------------------------
if "Vola" in filters:
    if "EW" in allocators:
        BacktesterObj = WFBacktester(filter=VolaFilter(rel_vola_min=0.2, rel_vola_max=0.4, min_close_price=min_close_price),
                                     allocator=EWAllocator(max_weight=max_weight),
                                     realloc_period=realloc_period, output_path="./output/Backtests/Vola+EW/")
        BacktesterObj.run(df_test)
        equity.append(pd.DataFrame(BacktesterObj.equity_.values,
                      index=BacktesterObj.equity_.index,
                      columns=["VolaFilter + EW"]))

    if "HRP" in allocators:
        # Vola filter, HRP
        BacktesterObj = WFBacktester(filter=VolaFilter(rel_vola_min=0.2, rel_vola_max=0.4, min_close_price=min_close_price),
                                     allocator=HRPAllocator(ret_mat=ret_mat, max_weight=max_weight, ret_window=ret_window),
                                     realloc_period=realloc_period, output_path="./output/Backtests/Vola+HRP/")
        BacktesterObj.run(df_test)
        equity.append(pd.DataFrame(BacktesterObj.equity_.values,
                      index=BacktesterObj.equity_.index,
                      columns=["VolaFilter + HRP"]))

# ML filter, Equal Weight ----------------------------------------------------------------------------------------------
if "ML" in filters:
    if "EW" in allocators:
        BacktesterObj = WFBacktester(filter=MLFilter(models, df, feature_list, labels, purge_window=120,
                                                     up_proba=up_proba, realloc_nb_til_retrain=realloc_nb_til_retrain,
                                                     min_close_price=min_close_price, max_rel_size=max_rel_size),
                                     allocator=EWAllocator(max_weight=max_weight),
                                     realloc_period=realloc_period, output_path="./output/Backtests/ML+EW/")
        BacktesterObj.run(df_test)
        equity.append(pd.DataFrame(BacktesterObj.equity_.values,
                      index=BacktesterObj.equity_.index,
                      columns=["MLFilter + EW"]))

    if "HRP" in allocators:
        # ML filter, HRP
        BacktesterObj = WFBacktester(filter=MLFilter(models, df, feature_list, labels, purge_window=120,
                                                     up_proba=up_proba, realloc_nb_til_retrain=realloc_nb_til_retrain,
                                                     min_close_price=min_close_price, max_rel_size=max_rel_size),
                                     allocator=HRPAllocator(ret_mat=ret_mat, max_weight=max_weight, ret_window=ret_window),
                                     realloc_period=realloc_period, output_path="./output/Backtests/ML+HRP/")
        BacktesterObj.run(df_test)
        equity.append(pd.DataFrame(BacktesterObj.equity_.values,
                      index=BacktesterObj.equity_.index,
                      columns=["MLFilter + HRP"]))

# save equity curves
equity = pd.concat(equity, axis=1)
equity.index.name = "Date"
equity.to_csv("./output/Backtests/Equity_df.csv", sep=";")

# plot equity
equity = pd.melt(equity.reset_index(), id_vars="Date", var_name="Method", value_name="Equity")

fig, ax = plt.subplots(figsize=(16, 8))
plt.grid(which="minor")
ax.yaxis.set_major_locator(MultipleLocator(1000))
ax.set_yscale("log")
# ax.set(yscale="log")
# ax.xaxis.set_major_locator(mdates.YearLocator())
sns.set(style="whitegrid", font_scale=2)

sns.lineplot(x="Date", y="Equity", data=equity, hue="Method")
plt.savefig("./output/Backtests/Equity_plot.png")

end_time = time.time()
print("Runtime: ", time.strftime('%H:%M:%S', time.gmtime(end_time - start_time)))
