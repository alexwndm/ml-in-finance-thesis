# %% XGBoost Classifier

import xgboost as xgb
from classes.ReturnPredictor import RPClassifier


class RPXGBoostClassifier(RPClassifier):
    def __init__(self, params=None, **kwargs):
        if params is not None:
            # set default values
            if "max_depth" not in params:
                params["max_depth"] = 5
            if "n_jobs" not in params:
                params["n_jobs"] = -1
            if "random_state" not in params:
                params["random_state"] = 0
            model = xgb.XGBClassifier(**params)
            super().__init__(model)
        else:
            # set default values
            if "max_depth" not in kwargs:
                kwargs["max_depth"] = 10
            if "n_jobs" not in kwargs:
                kwargs["n_jobs"] = -1
            if "random_state" not in kwargs:
                kwargs["random_state"] = 0
            model = xgb.XGBClassifier(**kwargs)
            super().__init__(model)
        self.model_class = "BoostedTreeClassifier"




