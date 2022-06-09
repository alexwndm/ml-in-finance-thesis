# Filter top assets for a given day

# general
import pandas as pd

from functions.train_test_splits import get_train_test_indices


# return every asset that currently is part of SP500
class NoFilter:
    def __init__(self, min_close_price=1):
        self.method = "NoFilter"
        self.min_close_price = min_close_price

    def choose_assets(self, curr_df):  # SP500 member only, no penny stocks
        return curr_df.loc[(curr_df.IndexMember >= 1) & (curr_df.CloseAdj >= self.min_close_price),
                           "AssetSymbol"].unique()


# return assets in vola basket
class VolaFilter:
    def __init__(self, rel_vola_min=0.2, rel_vola_max=0.4, min_close_price=1, underlying_vola="Vola250"):
        self.method = "VolaFilter"
        self.rel_vola_min = rel_vola_min
        self.rel_vola_max = rel_vola_max
        self.min_close_price = min_close_price
        self.underlying_vola = underlying_vola

    def choose_assets(self, curr_df):
        filtered_df = curr_df.loc[(curr_df.IndexMember >= 1) & (curr_df.CloseAdj >= self.min_close_price)]  # SP500, no penny stock
        sorted_assets = filtered_df.sort_values(self.underlying_vola).AssetSymbol.values
        assets = sorted_assets[int(self.rel_vola_min * filtered_df.shape[0]):
                               int(self.rel_vola_max * filtered_df.shape[0])]
        return assets


# Use Machine Learning classifiers to select assets with best prediction
class MLFilter:
    def __init__(self, classifier_list, df, feature_list, label_list, up_proba=0.5, max_rel_size=None, min_close_price=1,
                 realloc_nb_til_retrain=4, purge_window=120):
        if type(classifier_list) is not list:
            self.classifier_list = [classifier_list]  # single clf
        else:
            self.classifier_list = classifier_list
        self.df = df
        self.feature_list = feature_list
        self.label_list = label_list
        self.method = "MLFilter"  #[clf.model_class + clf.label_ for clf in self.classifiers]
        self.up_proba = up_proba
        self.max_rel_size = max_rel_size
        self.min_close_price = min_close_price
        self.realloc_nb_til_retrain = realloc_nb_til_retrain  # nb of reallocations until train dataset is updated and clf refitted
        self.retrain_counter = realloc_nb_til_retrain  # model has to be fitted in beginning
        self.purge_window = purge_window

    # exclude classifiers, as they are too big to save. Used in WFBacktester.save_backtest/pickle.dump
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['classifier_list']
        return state

    # filter top assets for a single day
    def choose_assets(self, curr_df):

        # retrain?
        if self.retrain_counter >= self.realloc_nb_til_retrain:
            self.retrain_clf(test_border=curr_df.index[0])  # update train dataset, retrain classifier
            self.retrain_counter = 0
        else:
            self.retrain_counter += 1

        # index only, no penny stock
        filtered_df = curr_df.loc[(curr_df.IndexMember >= 1) & (curr_df.CloseAdj >= self.min_close_price)]

        y_pred_proba_df = pd.DataFrame(data={"AssetSymbol": filtered_df["AssetSymbol"]})
        for clf, features in zip(self.classifier_list, self.feature_list):
            up_pred = pd.Series(clf.predict_proba(filtered_df[features])[:, 1], name=clf.label_,
                                index=filtered_df.index)
            y_pred_proba_df = pd.concat([y_pred_proba_df, up_pred], axis=1)

        y_pred_proba_df["ClfMean"] = y_pred_proba_df.mean(axis=1)
        # y_pred_proba_df["sd"] = y_pred_proba_df.std(axis=1)

        top_assets = y_pred_proba_df.loc[y_pred_proba_df["ClfMean"] >= self.up_proba]
        print("Number of assets with p over %.2f: %d of %d." %
              (self.up_proba, top_assets.shape[0], y_pred_proba_df.shape[0]))
        if self.max_rel_size is not None:
            size_aim = int(filtered_df.shape[0] * self.max_rel_size)
            top_assets = top_assets.nlargest(n=size_aim, columns="ClfMean")

        # df0 = y_pred_proba_df.describe().iloc[[1, 2]]
        # df1 = y_pred_proba_df[y_pred_proba_df.AssetSymbol == "AAPL_US"].copy(deep=True)
        # df1.index = ["AAPL"]
        # df1.drop("AssetSymbol", axis=1, inplace=True)
        # print(pd.concat([df0, df1], axis=0))
        # # print(df0)

        return top_assets["AssetSymbol"].values

    # fit classifiers
    def retrain_clf(self, test_border):
        train_indices, _ = get_train_test_indices(self.df.index, test_border=test_border,
                                                  purge_window=self.purge_window)
        df_train = self.df.iloc[train_indices].dropna()

        for clf, label, features in zip(self.classifier_list, self.label_list, self.feature_list):
            print("Fitting %s with %s." % (clf.model_class, label))
            clf.fit(df_train[features], df_train[label])

        return


