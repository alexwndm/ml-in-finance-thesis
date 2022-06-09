# %% initialization ====================================================================================================

# general
import pandas as pd
import time
import sys
import os

# plots
from functions.plots import plot_heatmap

# other
from functions.data_manipulation import read_data, read_index_data, export_features
from functions.train_test_splits import get_train_test_indices
from functions.hyperparameter_tuning import get_best_params


from classes.ReturnPredictor import load_model
from classes.DecisionTreeClassifier import RPDecisionTreeClassifier
from classes.RandomForestClassifier import RPRandomForestClassifier
from classes.XGBoostClassifier import RPXGBoostClassifier
import features as f


# %% main ==============================================================================================================

# parameters -----------------------------------------------------------------------------------------------------------
# part 1: data processing ----------
# skip_data_processing = True
skip_data_processing = False

# if not skip, output file:
output_file = "./data/TestDatasetSP.csv"
# set parameters
nb_of_assets = 30  # int or None
label_type = "edge_ratio_return"
return_horizons = [10, 25, 50, 100, 150, 200, 250]
label_horizons = [60, 90, 120]
index_rolling_windows = [10, 25, 50, 100, 150, 200, 250]
index = "SP500"
# index = "STOXX600"

# skip, input file?
# input_file = "SPMrkt.csv"
# input_file = "TestDatasetSTOXX.csv"
input_file = "TestDatasetSP.csv"


# part 2: data analysis ----------
# skip_data_analysis = True
skip_data_analysis = False

# features and exact label have to be selected later

# relative test size
test_size = 0.5

# choose models
models = ["RF"]  # DT, RF or BT

# CV hyperparameter tuning on the training dataset
tune_hyperparameters = False

# final output: model scores, feature importance, confusion matrix

# read data ------------------------------------------------------------------------------------------------------------
start_reading_data = time.time()

if not skip_data_processing:
    print("Reading data.")
    print("Index:", index)

    df = read_data("./data/Marktdaten/" + index + "TR/Equities_clean", index=index, nb_of_assets=nb_of_assets,
                   label_type=label_type, value="CloseAdj",
                   return_horizons=return_horizons, label_horizons=label_horizons)
    print("Reading index data.")
    mrkt_df = read_index_data("./data/" + index + "TR.csv", vix_filepath="./data/VIX.csv",
                              label_type=label_type,
                              return_horizons=return_horizons, rolling_windows=index_rolling_windows)
    df = df.join(mrkt_df, how="inner")
    # add market residualized returns
    for window in return_horizons:
        df["ReturnIndexRatio" + str(window)] = df["Return" + str(window)] / df["IndexReturn" + str(window)].replace(0, 0.01)
        df["ReturnIndexRes" + str(window)] = df["Return" + str(window)] - df["IndexReturn" + str(window)]

    # mrkt_features = [f for f in df.columns if f[:4] == "Mrkt"]
    # df = df[df.AssetSymbol == "XOM_US"]
    # df = df[mrkt_features]
    # export features
    export_features(df)
    # export dataset
    df.to_csv(output_file)
else:
    print("Using %s." % input_file)
    df = pd.read_csv("./data/" + input_file, index_col="Date", parse_dates=True)
    export_features(df, title="OldFeatures")

end_reading_data = time.time()
print("Time to read data: ", time.strftime('%H:%M:%S', time.gmtime(end_reading_data - start_reading_data)))

if skip_data_analysis:
    print("Dataset exported.")
    sys.exit()

# plot data ------------------------------------------------------------------------------------------------------------

# year_distr = pd.Series(df.index.unique().year.values).value_counts().sort_index()

# drop NAs
df.dropna(inplace=True)
# year_distr_no_NA = pd.Series(df.index.unique().year.values).value_counts().sort_index()
df = df[df.IndexMember >= 1]
# year_distr_index_only = pd.Series(df.index.unique().year.values).value_counts().sort_index()

# year_distr = pd.concat([year_distr, year_distr_no_NA, year_distr_index_only], axis=1)
# year_distr.columns = ["Original", "DropNA", "IndexOnly"]

# print("Trading days per year in dataset: \n", year_distr)

# Feature Importance Test ----------------------------------------------------------------------------------------------

# market features
plot_heatmap(df[f.features_mrkt], title="Market Feature Correlation")

# to plot all heatmaps, the specific features have to be uncommented in data_manipulation/read_data.py

# plot_heatmap(df[f.features_indiv], title="Individual Feature Correlation")

# plot_heatmap(df[f.features_env], title="Adaptive Envelope Feature Correlation")

# plot_heatmap(df[f.features_boll], title="Adaptive Bollinger Feature Correlation")

# plot_heatmap(df[f.features_macd], title="Adaptive MACD Feature Correlation")

# plot_heatmap(df[f.features_osci], title="Adaptive Oscillator Feature Correlation")

# return features
plot_heatmap(df[f.features_return], title="Return Feature Correlation")

plot_heatmap(df[f.features_vgl], title="Feature Correlation")

# fix final features
features = f.features_good
# select label horizon
label_horizon = 120
# label_horizon = 60

label = ["Label" + str(label_horizon)]
# train test split for ML ----------------------------------------------------------------------------------------------

print("Start train test split.")
start_train_test_split = time.time()

# get train and test dates_unique
train_indices, test_indices = get_train_test_indices(df.index, test_size=test_size, purge_window=label_horizon)

# select data
X_train = df[features].iloc[train_indices]
y_train = df[label].iloc[train_indices]

X_test = df[features].iloc[test_indices]
y_test = df[label].iloc[test_indices]

print("Train from %s to %s." % (str(X_train.iloc[0].name)[:10], str(X_train.iloc[-1].name)[:10]))
print("Test from %s to %s." % (str(X_test.iloc[0].name)[:10], str(X_test.iloc[-1].name)[:10]))

end_train_test_split = time.time()
print("Time for train test split: ",
      time.strftime('%H:%M:%S', time.gmtime(end_train_test_split - start_train_test_split)))

# get optimal parameters -----------------------------------------------------------------------------------------------


if tune_hyperparameters:
    print("Start hyperparameter tuning.")

start_hyperparameter_tuning = time.time()
scoring = "precision"
# scoring = "neg_log_loss"

# Decision Tree ....................................................................................................
if "DT" in models:
    if tune_hyperparameters:
        params = {"criterion": ["entropy"], "max_depth": [8, None], "min_samples_split": [2, 10],
                  "max_features": [None, "sqrt"], "min_impurity_decrease": [0.0]}
        best_params = get_best_params(RPDecisionTreeClassifier, params, X=X_train, y=y_train, cv=5,
                                      scoring=scoring, purge_window=label_horizon, embargo_percentage=0.01)
        print("Best parameters are: ", best_params)
    else:
        best_params = {'criterion': 'entropy', 'max_depth': 3, 'min_samples_split': 10, 'max_features': None,
                       'min_impurity_decrease': 0.0, "class_weight": "balanced"}
        # best_params = {'criterion': 'gini', 'max_depth': None, 'min_samples_split': 10, 'max_features': 'sqrt',
        #                'min_impurity_decrease': 0.0}
    DT_model = RPDecisionTreeClassifier(best_params)

# Random Forest ....................................................................................................
if "RF" in models:
    if tune_hyperparameters:
        params = {"n_estimators": [100, 200], "criterion": ["entropy"],
                  "min_weight_fraction_leaf": [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07], "max_features": [1, 2, 3, 4, 5],
                  "class_weight": ["balanced_subsample"], "bootstrap": [True]}
        best_params = get_best_params(RPRandomForestClassifier, params, X=X_train, y=y_train, cv=5,
                                      scoring=scoring, purge_window=label_horizon, embargo_percentage=0.01)
        print("Best parameters are: ", best_params)
    else:
        # best_params = {'n_estimators': 200, 'criterion': 'entropy',
        #                "min_weight_fraction_leaf": 0.01,
        #                'max_features': 5, 'bootstrap': True, 'class_weight': 'balanced_subsample'}

        # for feature importance tests:
        best_params = {'n_estimators': 100, 'criterion': 'entropy',
                       'max_depth': len(features),
                       'max_features': 1, 'bootstrap': True, 'class_weight': 'balanced_subsample'}
    RF_model = RPRandomForestClassifier(best_params)

# Boosted Tree .....................................................................................................
if "BT" in models:
        if tune_hyperparameters:
            pos_label_count = sum(n > 0 for n in df[label].values)[0]  # class weights. For binary classes only
            params = {"max_depth": [3, 4, 5], "learning_rate": [0.05, 0.1], "n_estimators": [50, 100], "colsamlpe_bytree": [0.1, 0.5],
                      "subsample": [0.8], "eval_metric": ["logloss"], "gamma": [0.5],
                      "scale_pos_weight": [(df.shape[0] - pos_label_count) / pos_label_count]}

            best_params = get_best_params(RPXGBoostClassifier, params, X=X_train, y=y_train, cv=5,
                                          scoring=scoring, purge_window=label_horizon, embargo_percentage=0.01)
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

# Decision Tree ....................................................................................................
if "DT" in models:
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
if "RF" in models:
    if pre_fitted:
        RF_model = load_model("RandomForestClassifier" + label[0])

    start_fitting = time.time()

    RF_model.evaluate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scoring="all",
                      pre_fitted=pre_fitted, save_model=False)
    # RF_model.cv_score(X=df[features], y=df[label], cv=5, purge_window=label_horizon)

    end_fitting = time.time()
    print("Time to fit the %s: " % RF_model.model_class,
          time.strftime('%H:%M:%S', time.gmtime(end_fitting - start_fitting)))

    # RF_model.cv_score(X=df[features], y=df[label], cv=5)
# Boosted Tree .....................................................................................................
if "BT" in models:
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
