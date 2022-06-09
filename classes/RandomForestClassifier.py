# %% Random Forest Classifier

import numpy as np
import pandas as pd

from classes.ReturnPredictor import RPClassifier
from sklearn.ensemble import RandomForestClassifier


# Mean Decrease Impurity for when max_features=1.
# See Lopez, p. 115.
def featImpMDI(fit, featNames):
    # feat importance based on IS mean impurity reduction
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan)  # because max_features=1
    imp = pd.concat({'mean': df0.mean(), 'std': df0.std() * df0.shape[0] ** -.5}, axis=1)
    imp /= imp['mean'].sum()
    return imp


class RPRandomForestClassifier(RPClassifier):
    def __init__(self, params=None, **kwargs):
        if params is not None:  # parameters as dictionary
            # set default values
            if "n_jobs" not in params:
                params["n_jobs"] = -1
            if "random_state" not in params:
                params["random_state"] = 0
            model = RandomForestClassifier(**params)
            super().__init__(model)
        else:
            # set default values
            if "n_jobs" not in kwargs:
                kwargs["n_jobs"] = -1
            if "random_state" not in kwargs:
                kwargs["random_state"] = 0
            model = RandomForestClassifier(**kwargs)
            super().__init__(model)
        self.model_class = "RandomForestClassifier"

    def fit(self, X_train, y_train):
        super().fit(X_train, y_train)
        self.feature_importances_ = featImpMDI(fit=self.model, featNames=self.features_)  # feat imp with mean and std





