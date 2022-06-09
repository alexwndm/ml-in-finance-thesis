
import pandas as pd

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

# read data ------------------------------------------------------------------------------------------------------------
print("Read data.")
df = pd.read_csv("./data/TestDatasetSTOXX.csv", index_col="Date", parse_dates=True)
# print(df.columns)
# df.columns = ['CloseAdj', 'AssetSymbol', 'IndexMember', 'Week', 'Return1',
#        'Bollinger10', 'Envelope10', 'Donchian10', 'VolaEqW10', 'DD10',
#        'Grad10', 'Bollinger30', 'Envelope30', 'Donchian30', 'VolaEqW30',
#        'DD30', 'MACD30', 'Grad30', 'Bollinger60', 'Envelope60',
#        'Donchian60', 'VolaEqW60', 'DD60', 'MACD60', 'Grad60',
#        'Bollinger120', 'Envelope120', 'Donchian120', 'VolaEqW120', 'DD120',
#        'MACD120', 'Grad120', 'Bollinger250', 'Envelope250', 'Donchian250',
#        'VolaEqW250', 'DD250', 'MACD250', 'Grad250', 'BollingerTrend250',
#        'DonchianTrend250', 'Label90', 'Label120', 'MrktTrendBollinger250',
#        'MrktCTrendBollinger250', 'MrktTrendDonchian250', 'MrktCloseAdj',
#        'MrktReturn10', 'MrktReturn30', 'MrktReturn60', 'MrktReturn120',
#        'MrktReturn250', 'MrktBollinger10', 'MrktEnvelope10', 'MrktDonchian10',
#        'MrktVolaEqW10', 'MrktDD10', 'MrktGrad10', 'MrktBollinger50',
#        'MrktEnvelope50', 'MrktDonchian50', 'MrktVolaEqW50', 'MrktDD50',
#        'MrktMACD50', 'MrktGrad50', 'MrktBollinger100', 'MrktEnvelope100',
#        'MrktDonchian100', 'MrktVolaEqW100', 'MrktDD100', 'MrktMACD100',
#        'MrktGrad100', 'MrktBollinger200', 'MrktEnvelope200',
#        'MrktDonchian200', 'MrktVolaEqW200', 'MrktDD200', 'MrktMACD200',
#        'MrktGrad200', 'MrktBollinger250', 'MrktEnvelope250',
#        'MrktDonchian250', 'MrktVolaEqW250', 'MrktDD250', 'MrktMACD250',
#        'MrktGrad250', 'MrktBollingerTrend250', 'MrktDonchianTrend250',
#        'VIX', 'ReturnMrktRatio10', 'ReturnMrktRes10', 'ReturnMrktRatio30',
#        'ReturnMrktRes30', 'ReturnMrktRatio60', 'ReturnMrktRes60',
#        'ReturnMrktRatio120', 'ReturnMrktRes120', 'ReturnMrktRatio250',
#        'ReturnMrktRes250']
# df.to_csv("C:/Users/Danie/Desktop/MA_local/data/TestDatasetSP.csv")
# import sys
# sys.exit()

# for label in [label for label in df.columns if label[:5] == "Label"]:
#     print("Label: " + label)
#     print(df[label].value_counts(normalize=True))

# train test split -----------------------------------------------------------------------------------------------------
print("Splitting dataset.")
train_indices, test_indices = get_train_test_indices(df.index, test_size=0.5, purge_window=120)

# select data  
all_cols = ['CloseAdj', 'AssetSymbol', 'IndexMember', 'Day', 'Week', 'Year',
            'Return1', 'Return10', 'Return30', 'Return60', 'Return120', 'Return250',
            'Bollinger10', 'Envelope10', 'Donchian10', 'Vola10', 'DD10', 'Grad10',
            'Bollinger30', 'Envelope30', 'Donchian30', 'Vola30', 'DD30', 'MACD30', 'Grad30',
            'Bollinger60', 'Envelope60', 'Donchian60', 'Vola60', 'DD60', 'MACD60', 'Grad60',
            'Bollinger120', 'Envelope120', 'Donchian120', 'Vola120', 'DD120', 'MACD120', 'Grad120',
            'Bollinger250', 'Envelope250', 'Donchian250', 'Vola250', 'DD250', 'MACD250', 'Grad250',
            'BollingerTrend250', 'DonchianTrend250',
            'Label90', 'Label120',
            'MrktBollinger250', 'MrktBollingerC250', 'MrktDonchian250',
            'IndexCloseAdj', 'IndexReturn10', 'IndexReturn30', 'IndexReturn60', 'IndexReturn120', 'IndexReturn250',
            'IndexBollinger10', 'IndexEnvelope10', 'IndexDonchian10', 'IndexVola10', 'IndexDD10', 'IndexGrad10',
            'IndexBollinger50', 'IndexEnvelope50', 'IndexDonchian50', 'IndexVola50', 'IndexDD50', 'IndexMACD50', 'IndexGrad50',
            'IndexBollinger100', 'IndexEnvelope100', 'IndexDonchian100', 'IndexVola100', 'IndexDD100', 'IndexMACD100', 'IndexGrad100',
            'IndexBollinger200', 'IndexEnvelope200', 'IndexDonchian200', 'IndexVola200', 'IndexDD200', 'IndexMACD200', 'IndexGrad200',
            'IndexBollinger250', 'IndexEnvelope250', 'IndexDonchian250', 'IndexVola250', 'IndexDD250', 'IndexMACD250', 'IndexGrad250',
            'IndexBollingerTrend250', 'IndexDonchianTrend250',
            'VIX',
            'ReturnIndexRatio10', 'ReturnIndexRes10',
            'ReturnIndexRatio30', 'ReturnIndexRes30',
            'ReturnIndexRatio60', 'ReturnIndexRes60',
            'ReturnIndexRatio120', 'ReturnIndexRes120',
            'ReturnIndexRatio250', 'ReturnIndexRes250']

features = ['CloseAdj', 'DayOfYear',
            'Bollinger60', 'Envelope60', 'Donchian60', 'Vola60', 'DD60', 'MACD60', 'Grad60',
            'Bollinger120', 'Envelope120', 'Donchian120', 'Vola120', 'DD120', 'MACD120', 'Grad120',
            'Bollinger250', 'Envelope250', 'Donchian250', 'Vola250', 'DD250', 'MACD250', 'Grad250',
            'MrktBollinger250', 'MrktBollingerC250', 'MrktDonchian250',
            'IndexCloseAdj',  'IndexReturn120', 'IndexReturn250',
            'IndexBollinger100', 'IndexEnvelope100', 'IndexDonchian100', 'IndexVola100', 'IndexDD100', 'IndexMACD100', 'IndexGrad100',
            'IndexBollinger200', 'IndexEnvelope200', 'IndexDonchian200', 'IndexVola200', 'IndexDD200', 'IndexMACD200', 'IndexGrad200',
            'IndexBollinger250', 'IndexEnvelope250', 'IndexDonchian250', 'IndexVola250', 'IndexDD250', 'IndexMACD250', 'IndexGrad250',
            'VIX',
            'ReturnIndexRatio60', 'ReturnIndexRes60',
            'ReturnIndexRatio120', 'ReturnIndexRes120',
            'ReturnIndexRatio250', 'ReturnIndexRes250']
featuresNlabels = features + ["AssetSymbol", "Return1", 'IndexMember', "Label90", "Label120"]
df = df[featuresNlabels]  # only use necessary data (for RAM..)

# features = list(df.drop([label for label in df.columns if label[:5] == "Label"]
#                         # + [ret for ret in df.columns if ret[:11] == "Return"]
#                         # + [phase for phase in df.columns if phase[:5] == "Phase" or phase[:9] == "MrktPhase"]
#                         + ["AssetSymbol"],
#                         axis=1).columns.values)

# specify models for MLFilter
models = list()
labels = list()
# Random Forest 1 ---------- TODO  200 300 100 n_est
best_params = {'n_estimators': 200, 'criterion': 'gini', 'max_depth': 30, 'min_samples_split': 2,
               'max_features': 1, 'max_leaf_nodes': None, 'bootstrap': True, 'oob_score': True,
               'class_weight': 'balanced_subsample'}
models.append(RPRandomForestClassifier(best_params))
labels.append(["Label120"])
# RF 2 ---------------------
best_params = {'n_estimators': 300, 'criterion': 'entropy', 'max_depth': 20, 'min_samples_split': 0.01,
               'max_features': 2, 'max_leaf_nodes': None, 'bootstrap': True, 'oob_score': True,
               'class_weight': 'balanced_subsample'}
models.append(RPRandomForestClassifier(best_params))
labels.append(["Label90"])
# RF 3 ---------------------
best_params = {'n_estimators': 100, 'criterion': 'gini', 'min_weight_fraction_leaf': 0.05, 'min_samples_split': 2,
               'max_features': 4, 'max_leaf_nodes': None, 'bootstrap': True, 'oob_score': True,
               'class_weight': 'balanced_subsample'}
# best_params = {'n_estimators': 100, 'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 2,
#                'max_features': 4, 'max_leaf_nodes': None, 'bootstrap': True, 'oob_score': True,
#                'class_weight': 'balanced_subsample'}
models.append(RPRandomForestClassifier(best_params))
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
max_weight = 0.05
ret_window = 5
realloc_period = 60
# No filter, Equal Weight ----------------------------------------------------------------------------------------------
BacktesterObj = WFBacktester(filter=NoFilter(min_close_price=min_close_price),
                             allocator=EWAllocator(max_weight=max_weight),
                             realloc_period=realloc_period, output_path="./output/Backtests/EW/")
BacktesterObj.run(df_test)
equity.append(pd.DataFrame(BacktesterObj.equity_.values,
              index=BacktesterObj.equity_.index,
              columns=["NoFilter + EW"]))

# No filter, HRP
BacktesterObj = WFBacktester(filter=NoFilter(min_close_price=min_close_price),
                             allocator=HRPAllocator(ret_mat=ret_mat, max_weight=max_weight, ret_window=ret_window),
                             realloc_period=realloc_period, output_path="./output/Backtests/HRP/")
BacktesterObj.run(df_test)
equity.append(pd.DataFrame(BacktesterObj.equity_.values,
              index=BacktesterObj.equity_.index,
              columns=["NoFilter + HRP"]))

# Vola filter, Equal Weight --------------------------------------------------------------------------------------------
BacktesterObj = WFBacktester(filter=VolaFilter(rel_vola_min=0.2, rel_vola_max=0.4, min_close_price=min_close_price),
                             allocator=EWAllocator(max_weight=max_weight),
                             realloc_period=realloc_period, output_path="./output/Backtests/Vola+EW/")
BacktesterObj.run(df_test)
equity.append(pd.DataFrame(BacktesterObj.equity_.values,
              index=BacktesterObj.equity_.index,
              columns=["VolaFilter + EW"]))

# Vola filter, HRP
BacktesterObj = WFBacktester(filter=VolaFilter(rel_vola_min=0.2, rel_vola_max=0.4, min_close_price=min_close_price),
                             allocator=HRPAllocator(ret_mat=ret_mat, max_weight=max_weight, ret_window=ret_window),
                             realloc_period=realloc_period, output_path="./output/Backtests/Vola+HRP/")
BacktesterObj.run(df_test)
equity.append(pd.DataFrame(BacktesterObj.equity_.values,
              index=BacktesterObj.equity_.index,
              columns=["VolaFilter + HRP"]))

# ML filter, Equal Weight ----------------------------------------------------------------------------------------------
BacktesterObj = WFBacktester(filter=MLFilter(models, df, features, labels, up_proba=0.55, purge_window=120,
                                             min_close_price=min_close_price, max_rel_size=0.2),
                             allocator=EWAllocator(max_weight=max_weight),
                             realloc_period=realloc_period, output_path="./output/Backtests/ML+EW/")
BacktesterObj.run(df_test)
equity.append(pd.DataFrame(BacktesterObj.equity_.values,
              index=BacktesterObj.equity_.index,
              columns=["MLFilter + EW"]))

# ML filter, HRP
BacktesterObj = WFBacktester(filter=MLFilter(models, df, features, labels, up_proba=0.55, purge_window=120,
                                             min_close_price=min_close_price, max_rel_size=0.2),
                             allocator=HRPAllocator(ret_mat=ret_mat, max_weight=max_weight, ret_window=ret_window),
                             realloc_period=realloc_period, output_path="./output/Backtests/ML+HRP/")
BacktesterObj.run(df_test)
equity.append(pd.DataFrame(BacktesterObj.equity_.values,
              index=BacktesterObj.equity_.index,
              columns=["MLFilter + HRP"]))






# save equity curves
equity = pd.concat(equity, axis=1)
equity.index.name = "Date"
equity.to_csv("./output/Backtests/Equity_df.csv")

# plot equity
equity = pd.melt(equity.reset_index(), id_vars="Date", var_name="Method", value_name="Equity")

fig, ax = plt.subplots(figsize=(15, 8))
sns.set(style="whitegrid")
plt.grid(which="minor")
ax.yaxis.set_major_locator(MultipleLocator(1000))
ax.set_yscale("log")
# ax.set(yscale="log")
ax.xaxis.set_major_locator(mdates.YearLocator())

sns.lineplot(x="Date", y="Equity", data=equity, hue="Method")
plt.savefig("./output/Backtests/Equity_plot.png")
