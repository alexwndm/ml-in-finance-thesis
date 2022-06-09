# Base ReturnPredictor class

import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, log_loss
from sklearn.utils.class_weight import compute_class_weight
from functions.train_test_splits import get_cv_dates


class ReturnPredictor:
    def __init__(self, model):
        self.model = model
        self.model_class = None

    def fit(self, X_train, y_train):
        self.model = self.model.fit(X_train.values, y_train.values.ravel())
        self.features_ = X_train.columns
        self.label_ = y_train.columns.values[0]
        self.feature_importances_ = pd.DataFrame(self.model.feature_importances_, index=self.features_,
                                                 columns=["feature_importances"])

    def predict(self, X_test):
        return self.model.predict(X_test.values)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test.values)

    # fit and evalate model on dataset
    def evaluate(self, X_train, y_train, X_test, y_test, scoring="all", pre_fitted=False, save_model=True):
        if not pre_fitted:  # fit model
            print("Fitting %s." % self.model_class)
            self.fit(X_train=X_train, y_train=y_train)
            if save_model:
                self.save_model()
        y_pred = self.predict(X_test)

        stats_file = open("./output/MLModels/%s_scores.txt" % self.model_class, "w")
        stats_file.write("%s \n" % self.model_class)
        if scoring == "all":
            # Accuracy
            acc = accuracy_score(y_test, y_pred)
            print("Accuracy: ", acc)
            stats_file.write("Accuracy: %.16f\n" % acc)

            # Precision
            prcn = precision_score(y_test, y_pred)
            print("Precision: ", prcn)
            stats_file.write("Precision: %.16f\n" % prcn)

            # F1 score
            f1 = f1_score(y_test, y_pred, average="macro")
            print("F1-Score: ", f1)
            stats_file.write("F1-Score: %.16f\n" % f1)

            # Neg log loss
            y_pred_proba = self.predict_proba(X_test)
            score = - log_loss(y_test, y_pred_proba)
            print("Negative log loss: ", score)
            stats_file.write("Negative log loss: %.16f\n" % score)
        elif scoring == "f1_score":
            score = f1_score(y_test, y_pred, average="macro")
            print("F1-Score: ", score)
            stats_file.write("F1-Score: %.16f\n" % score)
        elif scoring == "neg_log_loss":
            y_pred_proba = self.predict_proba(X_test)
            score = - log_loss(y_test, y_pred_proba)
            print("Negative log loss:", score)
            stats_file.write("Negative log loss: %.16f\n" % score)
        else:  # Acc
            score = accuracy_score(y_test, y_pred)
            print("Using Accuracy score. Accuracy: ", score)
            stats_file.write("Accuracy: %.16f\n" % score)
        stats_file.close()

        self.plot_importances()
        self.plot_confusion_matrix(y_test, y_pred)
        return score

    # Fit and score the model, using a cross validation scheme.
    def cv_score(self, X, y, cv=5, scoring="neg_log_loss", purge_window=10, embargo_percentage=0.01):
        cv_dates = get_cv_dates(dates=X.index, cv=cv, purge_window=purge_window, embargo_percentage=embargo_percentage)
        scores = []
        for fold in range(1, cv + 1):
            # print("Fitting model in fold %d of %d. Testing from %s to %s."
            #       % (fold, cv, X.iloc[cv_dates[fold - 1]["test_dates"].start].name.strftime("%Y-%m-%d"),
            #          X.iloc[cv_dates[fold - 1]["test_dates"].stop - 2].name.strftime("%Y-%m-%d")))
            X_train = X.iloc[cv_dates[fold - 1]["train_dates"]]
            y_train = y.iloc[cv_dates[fold - 1]["train_dates"]]

            X_test = X.iloc[cv_dates[fold - 1]["test_dates"]]
            y_test = y.iloc[cv_dates[fold - 1]["test_dates"]]

            self.fit(X_train=X_train, y_train=y_train)
            # self.evaluate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scoring="all")

            if scoring == "accuracy":
                y_pred = self.predict(X_test)
                scores.append(accuracy_score(y_test, y_pred))
            elif scoring == "precision":
                y_pred = self.predict(X_test)
                scores.append(precision_score(y_test, y_pred))
            elif scoring == "f1_score":
                y_pred = self.predict(X_test)
                scores.append(f1_score(y_test, y_pred, average="macro"))
            else:  # neg log loss
                y_pred_proba = self.predict_proba(X_test)
                # class_weights = compute_class_weight("balanced", classes=np.unique(y_train.values.flatten()),
                #                                      y=y_train.values.flatten())
                # sample_weights = np.apply_along_axis(np.vectorize(lambda y:
                #                                                   class_weights[0] if y < 0.5 else class_weights[1]),
                #                                      0, y_pred_proba[:, 1])
                scores.append(- log_loss(y_test, y_pred_proba))
                # scores.append(- log_loss(y_test, y_pred_proba, sample_weight=sample_weights))
            # print("Score: ", scores[-1])
        print("Average score is %.4f +/- %.4f." % (np.mean(scores), np.std(scores)))
        return scores

    def save_model(self, filepath="./output/MLModels/Models/"):
        filepath = filepath + self.model_class + self.label_
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    # plots ------------------------------------------------------------------------------------------------------------
    # plot importances
    def plot_importances(self, title="Feature Importance, 120-day Edge Ratio Label"):  #TODO
        importances = self.feature_importances_.iloc[:, 0]
        importances = importances.sort_values(ascending=False)
        if len(importances) <= 30:  # fit features in one plot
            sns.barplot(importances.values, importances.index).set_title(title)
            # plt.subplots_adjust(top=0.925, bottom=0.079, left=0.356, right=0.977, hspace=0.2, wspace=0.2)
            plt.tight_layout()
            plt.savefig('./output/MLModels/%s_feature_importances.png' % self.model_class)
            plt.show()
        else:  # use at least 2 plots
            end = 30  # first plot
            sns.barplot(importances.values[:end], importances.index[:end]).set_title(title + " (best)")
            _, xmax, _, _ = plt.axis()
            plt.subplots_adjust(top=0.925, bottom=0.079, left=0.356, right=0.977, hspace=0.2, wspace=0.2)
            plt.savefig('./output/MLModels/%s_feature_importances%s.png' % (self.model_class, "_best"))
            plt.show()
            if len(importances) > 60:  # 3 plots required, add mid plot
                end = 60
                sns.barplot(importances.values[30:60], importances.index[30:60]).set_title(title + " (mid)")
                plt.xlim(0, xmax)
                plt.subplots_adjust(top=0.925, bottom=0.079, left=0.356, right=0.977, hspace=0.2, wspace=0.2)
                plt.savefig('./output/MLModels/%s_feature_importances%s.png' % (self.model_class, "_mid"))
                plt.show()
            if len(importances) > 90:  # 4 plots required, add 2nd mid plot
                end = 90
                sns.barplot(importances.values[60:90], importances.index[60:90]).set_title(title + " (mid2)")
                plt.xlim(0, xmax)
                plt.subplots_adjust(top=0.925, bottom=0.079, left=0.356, right=0.977, hspace=0.2, wspace=0.2)
                plt.savefig('./output/MLModels/%s_feature_importances%s.png' % (self.model_class, "_mid2"))
                plt.show()
            # final plot
            sns.barplot(importances.values[end:], importances.index[end:]).set_title(title + " (worst)")
            plt.xlim(0, xmax)
            plt.subplots_adjust(top=0.925, bottom=0.079, left=0.356, right=0.977, hspace=0.2, wspace=0.2)
            plt.savefig('./output/MLModels/%s_feature_importances%s.png' % (self.model_class, "_worst"))
            plt.show()

    # plot confusion matrix
    def plot_confusion_matrix(self, y_test, y_pred, normalize=False, title="Confusion Matrix"):
        # get confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        # use classes to create pd dataframe
        classes = np.unique(y_test)
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)

        # plot confusion matrix
        #    fig, ax = plt.subplots(figsize=(7,7))
        fmt = ".4f" if normalize else "d"
        ax = sns.heatmap(cm_df, fmt=fmt,
                         cmap=plt.cm.Blues,
                         cbar=True, annot=True, square=True)
        # sns.heatmap broke in matplotlib 3.1.1.. Adjust plot manually
        #    import matplotlib
        #    print(matplotlib.__version__)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title(title)
        plt.savefig('./output/MLModels/%s_confusion_matrix.png' % self.model_class)
        plt.show()


class RPClassifier(ReturnPredictor):
    def __init__(self, model):
        super().__init__(model)

    def fit(self, X_train, y_train):
        super().fit(X_train, y_train)
        self.classes_ = np.unique(y_train.values).astype("str")


class RPRegressor(ReturnPredictor):
    def __init__(self, model):
        super().__init__(model)


# other ----------------------------------------------------------------------------------------------------------------
# load model
def load_model(file, filepath="./output/MLModels/Models/"):
    filepath = filepath + file
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model


