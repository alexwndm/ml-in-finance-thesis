# %% initialization ====================================================================================================

# general
import pandas as pd
import time
import sys

# plots
from functions.plots import plot_heatmap

# other
from functions.data_manipulation import read_data, read_index_data
from functions.train_test_splits import get_train_test_indices
from functions.hyperparameter_tuning import get_best_params


from classes.ReturnPredictor import load_model
from classes.DecisionTreeClassifier import RPDecisionTreeClassifier
from classes.RandomForestClassifier import RPRandomForestClassifier
from classes.XGBoostClassifier import RPXGBoostClassifier

# %% main ==============================================================================================================

read_data_only = False
# read data ------------------------------------------------------------------------------------------------------------
start_reading_data = time.time()

skip_data_manipulation = True

if not skip_data_manipulation:
    print("Reading data.")
    # set parameters
    nb_of_assets = 300 # int or None
    label_type = "edge_ratio_return"
    vola_weights = "equal_weights"
    return_horizons = [10, 30, 60, 120, 250]
    label_horizons = [90, 120]
    mrkt_rolling_windows = [10, 50, 100, 200, 250]
    index = "STOXX600"
    print("Index:", index)

    df = read_data("./data/Marktdaten/" + index + "TR/Equities_clean", index=index, nb_of_assets=nb_of_assets,
                   label_type=label_type, vola_weights=vola_weights, value="CloseAdj",
                   return_horizons=return_horizons, label_horizons=label_horizons)
    print("Reading market data.")
    mrkt_df = read_index_data("./data/" + index + "TR.csv", vix_filepath="./data/VIX.csv",
                              label_type=label_type, vola_weights=vola_weights,
                              return_horizons=return_horizons, rolling_windows=mrkt_rolling_windows)
    df = df.join(mrkt_df, how="inner")
    # add market residualized returns
    for window in return_horizons:
        df["ReturnIndexRatio" + str(window)] = df["Return" + str(window)] / df["IndexReturn" + str(window)].replace(0, 0.01)
        df["ReturnIndexRes" + str(window)] = df["Return" + str(window)] - df["IndexReturn" + str(window)]

    # drop unnecessary features after they have been used for other features
    df.drop(#["Return" + str(h) for h in return_horizons] +
            ["IndexReturn1"],
            axis=1, inplace=True)
    print([col for col in df.columns])
    df.to_csv("./data/TestDataset.csv")
else:
    print("Using TestDataset.")
    df = pd.read_csv("./data/TestDataset300.csv", index_col="Date", parse_dates=True)

end_reading_data = time.time()
print("Time to read data: ", time.strftime('%H:%M:%S', time.gmtime(end_reading_data - start_reading_data)))

if read_data_only:
    print("Dataset exported.")
    sys.exit()

# plot data ------------------------------------------------------------------------------------------------------------


# from functions.plots import *
# plot_data(df)
# plot_single_asset(df[df["AssetSymbol"] == df["AssetSymbol"].unique()[1]])

# drop NAs
df.dropna(inplace=True)
df = df[df.IndexMember >= 1]

# plot feature correlations
return_features = ['Return10', 'Return30', 'Return60', 'Return120', 'Return250',
                   'IndexReturn10', 'IndexReturn30', 'IndexReturn60', 'IndexReturn120', 'IndexReturn250',
                   'ReturnIndexRes10', 'ReturnIndexRes30', 'ReturnIndexRes60', 'ReturnIndexRes120', 'ReturnIndexRes250',
                   'ReturnIndexRatio10', 'ReturnIndexRatio30', 'ReturnIndexRatio60', 'ReturnIndexRatio120', 'ReturnIndexRatio250'
                   ]
# return_features = [ret for ret in df.columns if ret[:6] == "Return" or ret[:10] == "MrktReturn"]
plot_heatmap(df[return_features], title="Return Feature Correlation")


mrkt_features = ["IndexCloseAdj",
                 "IndexReturn250", "IndexEnvelope250", "IndexBollinger250", "IndexDonchian250",  "IndexMACD250", "IndexGrad250",
                 "MrktBollinger250", "MrktDonchian250",
                 "IndexVola250", "IndexDD250", "VIX"]
# mrkt_features = ["VIX", "MrktCloseAdj"] + [feature for feature in df.columns if feature[:4] == "Mrkt" and feature[-3:] == "250"]
plot_heatmap(df[mrkt_features], title="Market Feature Correlation")


raw_features = ['CloseAdj',  'Day', 'Week', 'Year',
                # 'Bollinger10', 'Envelope10', 'Donchian10', 'Vola10', 'DD10', 'Grad10',
                # 'Bollinger30', 'Envelope30', 'Donchian30', 'Vola30', 'DD30', 'MACD30', 'Grad30',
                # 'Bollinger60', 'Envelope60', 'Donchian60', 'Vola60', 'DD60', 'MACD60', 'Grad60',
                # 'Bollinger120', 'Envelope120', 'Donchian120', 'Vola120', 'DD120', 'MACD120', 'Grad120',
                'Envelope250', 'Bollinger250', 'Donchian250', 'MACD250', 'Grad250', 'Vola250', 'DD250']
# raw_features = list(df.drop([label for label in df.columns if label[:5] == "Label"]
#                             + ["AssetSymbol", "IndexMember"] + return_features
#                             # + [feature for feature in df.columns if feature[-2:] == "20"
#                             #    or feature[-3:] == "100" or feature[-3:] == "200" or feature[-2:] == "10"
#                             #    or feature[-1:] == "2" or feature[-1:] == "1"]
#                             + [feature for feature in df.columns if feature[:4] == "Mrkt"] + ["VIX"],
#                             axis=1).columns.values)
plot_heatmap(df[raw_features])


# final_features = list(df.drop([label for label in df.columns if label[:5] == "Label"]
#                               # + ["BollingerTrend", "MrktBollingerTrend"]
#                               # + [feat for feat in df.columns if feat[-2:] == "60"]
#                               # + [phase for phase in df.columns if phase[:5] == "Phase" or phase[:9] == "MrktPhase"]
#                               + ["IndexMember"]
#                               + ["AssetSymbol", "Return1"],
#                               axis=1).columns.values)


label_horizon = 120
label = ["Label" + str(label_horizon)]

# Feature Importances Test
features_window = ['Return10', 'Return30', 'Return60', 'Return120', 'Return250',
                    'Bollinger10', 'Envelope10', 'Donchian10', 'Vola10', 'DD10', 'Grad10',
                    'Bollinger30', 'Envelope30', 'Donchian30', 'Vola30', 'DD30', 'MACD30', 'Grad30',
                    'Bollinger60', 'Envelope60', 'Donchian60', 'Vola60', 'DD60', 'MACD60', 'Grad60',
                    'Bollinger120', 'Envelope120', 'Donchian120', 'Vola120', 'DD120', 'MACD120', 'Grad120',
                    'Bollinger250', 'Envelope250', 'Donchian250', 'Vola250', 'DD250', 'MACD250', 'Grad250',
                    'IndexReturn10', 'IndexReturn30', 'IndexReturn60', 'IndexReturn120', 'IndexReturn250',
                    'IndexBollinger10', 'IndexEnvelope10', 'IndexDonchian10', 'IndexVola10', 'IndexDD10', 'IndexGrad10',
                    'IndexBollinger50', 'IndexEnvelope50', 'IndexDonchian50', 'IndexVola50', 'IndexDD50', 'IndexMACD50', 'IndexGrad50',
                    'IndexBollinger100', 'IndexEnvelope100', 'IndexDonchian100', 'IndexVola100', 'IndexDD100', 'IndexMACD100', 'IndexGrad100',
                    'IndexBollinger200', 'IndexEnvelope200', 'IndexDonchian200', 'IndexVola200', 'IndexDD200', 'IndexMACD200', 'IndexGrad200',
                    'IndexBollinger250', 'IndexEnvelope250', 'IndexDonchian250', 'IndexVola250', 'IndexDD250', 'IndexMACD250', 'IndexGrad250',
                    'IndexBollingerTrend250', 'IndexDonchianTrend250',
                    'ReturnIndexRatio10', 'ReturnIndexRes10',
                    'ReturnIndexRatio30', 'ReturnIndexRes30',
                    'ReturnIndexRatio60', 'ReturnIndexRes60',
                    'ReturnIndexRatio120', 'ReturnIndexRes120',
                    'ReturnIndexRatio250', 'ReturnIndexRes250']

features_mrkt = ['MrktBollinger250', 'MrktBollingerC250', 'MrktDonchian250',
                'IndexCloseAdj',  'IndexReturn250',
                'IndexBollinger250', 'IndexEnvelope250', 'IndexDonchian250', 'IndexVola250', 'IndexDD250', 'IndexMACD250', 'IndexGrad250',
                'IndexBollingerTrend250', 'IndexDonchianTrend250',
                'VIX',
                'ReturnIndexRatio250', 'ReturnIndexRes250']

features_general = ['CloseAdj', 'AssetSymbol', 'IndexMember', 'Day', 'Week', 'Year',
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

good_features = ['CloseAdj', 'Week',
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

# fix final features
features = features_window

# train test split for ML ----------------------------------------------------------------------------------------------

print("Start train test split.")
start_train_test_split = time.time()

# get train and test dates_unique
train_indices, test_indices = get_train_test_indices(df.index, test_size=0.5, purge_window=label_horizon)

# select data
X_train = df[features].iloc[train_indices]
y_train = df[label].iloc[train_indices]

X_test = df[features].iloc[test_indices]
y_test = df[label].iloc[test_indices]

print("Test start: ", X_test.iloc[0].name)

end_train_test_split = time.time()
print("Time for train test split: ",
      time.strftime('%H:%M:%S', time.gmtime(end_train_test_split - start_train_test_split)))

# get optimal parameters -----------------------------------------------------------------------------------------------


models = ["RF"]  # DT, RF or BT
check_params = False
if check_params:
    print("Start hyperparameter tuning.")

start_hyperparameter_tuning = time.time()

for model in models:
    # Decision Tree ....................................................................................................
    if model == "DT":
        if check_params:
            params = {"criterion": ["entropy"], "max_depth": [8, None], "min_samples_split": [2, 10],
                      "max_features": [None, "sqrt"], "min_impurity_decrease": [0.0]}
            best_params = get_best_params(RPDecisionTreeClassifier, params, X=X_train, y=y_train, cv=5,
                                          scoring="neg_log_loss", purge_window=label_horizon, embargo_percentage=0.01)
            print("Best parameters are: ", best_params)
        else:
            best_params = {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 10, 'max_features': None,
                           'min_impurity_decrease': 0.0, "class_weight": "balanced"}
            # best_params = {'criterion': 'gini', 'max_depth': None, 'min_samples_split': 10, 'max_features': 'sqrt',
            #                'min_impurity_decrease': 0.0}
        DT_model = RPDecisionTreeClassifier(best_params)

    # Random Forest ....................................................................................................
    if model == "RF":
        if check_params:
            params = {"n_estimators": [100, 200], "criterion": ["entropy"], "max_depth": [None],
                      "min_weight_fraction_leaf": [0, 0.03, 0.05], "max_features": [1, 2, 4],
                      "class_weight": ["balanced_subsample"], "bootstrap": [True]}
            best_params = get_best_params(RPRandomForestClassifier, params, X=X_train, y=y_train, cv=5,
                                          scoring="neg_log_loss", purge_window=label_horizon, embargo_percentage=0.01)
            print("Best parameters are: ", best_params)
        else:
            # best_params = {'n_estimators': 100, 'criterion': 'gini', 'max_depth': None, 'min_samples_split': 2,
            #                'max_features': 'auto', 'max_leaf_nodes': None, 'bootstrap': True, 'oob_score': False}
            # best_params = {'n_estimators': 500, 'criterion': 'entropy', 'max_depth': len(features),
            #                'min_samples_split': 2, 'max_features': 1, 'max_leaf_nodes': None, 'bootstrap': True,
            #                'oob_score': False, "class_weight": "balanced_subsample"}

            best_params = {'n_estimators': 100, 'criterion': 'entropy', 'min_samples_split': 2,
                           "min_weight_fraction_leaf": 0.05,
                           'max_features': 1, 'bootstrap': True, 'class_weight': 'balanced_subsample'}
        RF_model = RPRandomForestClassifier(best_params)

    # Boosted Tree .....................................................................................................
    if model == "BT":
        if check_params:
            pos_label_count = sum(n > 0 for n in df[label].values)[0]  # class weights. For binary classes only
            params = {"max_depth": [3, 4, 5], "learning_rate": [0.05, 0.1], "n_estimators": [50, 100], "colsamlpe_bytree": [0.1, 0.5],
                      "subsample": [0.8], "eval_metric": ["logloss"], "gamma": [0.5],
                      "scale_pos_weight": [(df.shape[0] - pos_label_count) / pos_label_count]}

            best_params = get_best_params(RPXGBoostClassifier, params, X=X_train, y=y_train, cv=5,
                                          scoring="neg_log_loss", purge_window=label_horizon, embargo_percentage=0.01)
            print("Best parameters are: ", best_params)
        else:
            pos_label_count = sum(n > 0 for n in df[label].values)[0]  # class weights. For binary classes only
            best_params = {'max_depth': 10, 'learning_rate': 0.1, 'n_estimators': 100, "importance_type": "gain",
                           "scale_pos_weight": (df.shape[0] - pos_label_count) / pos_label_count}
        BT_model = RPXGBoostClassifier(best_params)

end_hyperparameter_tuning = time.time()
print("Time for hyperparameter tuning: ",
      time.strftime('%H:%M:%S', time.gmtime(end_hyperparameter_tuning - start_hyperparameter_tuning)))


# fit final model ------------------------------------------------------------------------------------------------------

pre_fitted = False

if not pre_fitted:
    print("Start fitting final models.")
else:
    print("Start evaluation on pre-fitted model.")

for model in models:
    # Decision Tree ....................................................................................................
    if model == "DT":
        if pre_fitted:
            DT_model = load_model("DecisionTreeClassifier" + label[0])

        start_fitting = time.time()

        DT_model.evaluate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scoring="all",
                          pre_fitted=pre_fitted)

        end_fitting = time.time()
        print("Time to fit the %s: " % DT_model.model_class,
              time.strftime('%H:%M:%S', time.gmtime(end_fitting - start_fitting)))

        # DT_model.cv_score(X=df[features], y=df[label], cv=5)
    # Random Forest ....................................................................................................
    if model == "RF":
        if pre_fitted:
            RF_model = load_model("RandomForestClassifier" + label[0])

        start_fitting = time.time()

        RF_model.evaluate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scoring="all",
                          pre_fitted=pre_fitted)
        # RF_model.cv_score(X=df[features], y=df[label], cv=5, purge_window=label_horizon)

        end_fitting = time.time()
        print("Time to fit the %s: " % RF_model.model_class,
              time.strftime('%H:%M:%S', time.gmtime(end_fitting - start_fitting)))

        # RF_model.cv_score(X=df[features], y=df[label], cv=5)
    # Boosted Tree .....................................................................................................
    if model == "BT":
        if pre_fitted:
            BT_model = load_model("BoostedTreeClassifier" + label[0])

        start_fitting = time.time()

        BT_model.evaluate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scoring="all",
                          pre_fitted=pre_fitted)
        # BT_model.cv_score(X=df[features], y=df[label], cv=5, purge_window=label_horizon)

        end_fitting = time.time()
        print("Time to fit the %s: " % BT_model.model_class,
              time.strftime('%H:%M:%S', time.gmtime(end_fitting - start_fitting)))

        # BT_model.cv_score(X=df[features], y=df[label], cv=5)


# runtime summary ------------------------------------------------------------------------------------------------------

print("Time to read data: ", time.strftime('%H:%M:%S', time.gmtime(end_reading_data - start_reading_data)))
print("Time for train test split: ",
      time.strftime('%H:%M:%S', time.gmtime(end_train_test_split - start_train_test_split)))
print("Time for hyperparameter tuning: ",
      time.strftime('%H:%M:%S', time.gmtime(end_hyperparameter_tuning - start_hyperparameter_tuning)))
print("Complete time to run: ", time.strftime('%H:%M:%S', time.gmtime(time.time() - start_reading_data)))




