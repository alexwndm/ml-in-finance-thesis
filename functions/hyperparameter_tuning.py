# functions for hyperparameter tuning

# general
import numpy as np
from itertools import product


# return list of parameter dictionaries containing one value per parameter
def choose_params(params):
    keys = params.keys()
    vals = params.values()
    params_list = []
    for instance in product(*vals):
        params_list.append(dict(zip(keys, instance)))
    return params_list


# get best parameter combination for a given model, using cross validation
def get_best_params(model_class, params, X, y, cv=5, scoring="neg_log_loss", purge_window=10, embargo_percentage=0.01):
    params_list = choose_params(params)  # get a list of every parameter combination
    best_score = -np.inf
    best_score_std = np.inf
    best_params = params_list[0]
    counter = 1
    for param_combination in params_list:
        print("Testing parameter combination %d of %d." % (counter, len(params_list)))
        print(param_combination)
        model = model_class(**param_combination)
        scores = model.cv_score(X=X, y=y, cv=cv, scoring=scoring, purge_window=purge_window,
                                embargo_percentage=embargo_percentage)
        score = np.mean(scores)
        score_std = np.std(scores)
        if score > best_score:
            best_score = score
            best_score_std = score_std
            best_params = param_combination
        elif (best_score - score < 0.001) and (best_score_std > score_std):
            best_score = score
            best_score_std = score_std
            best_params = param_combination
        counter += 1

    print("Best score is %.4f +/- %.4f." % (best_score, best_score_std))
    return best_params

