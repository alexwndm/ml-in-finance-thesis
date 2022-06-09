# %% DecisionTree Classifier

import os
from subprocess import call

import matplotlib.pyplot as plt
from classes.ReturnPredictor import RPClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree


class RPDecisionTreeClassifier(RPClassifier):
    def __init__(self, params=None, **kwargs):
        if params is not None:  # parameters as dictionary
            # set default values
            if "max_depth" not in params:
                params["max_depth"] = 4
            if "criterion" not in params:
                params["criterion"] = "entropy"
            if "random_state" not in params:
                params["random_state"] = 0
            model = DecisionTreeClassifier(**params)
            super().__init__(model)
        else:
            # set default values
            if "max_depth" not in kwargs:
                kwargs["max_depth"] = 4
            if "criterion" not in kwargs:
                kwargs["criterion"] = "entropy"
            if "random_state" not in kwargs:
                kwargs["random_state"] = 0
            model = DecisionTreeClassifier(**kwargs)
            super().__init__(model)
        self.model_class = "DecisionTreeClassifier"

    def evaluate(self, X_train, y_train, X_test, y_test, scoring="f1_score", pre_fitted=False, save_model=True):
        super().evaluate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, scoring=scoring,
                         pre_fitted=False, save_model=True)
        self.plot_decision_tree()

    def plot_decision_tree(self, export_plot=True, proportion=True):
        if self.model.max_depth is None or self.model.max_depth > 5:
            print("The tree is too large to plot.")
            return None
        if not export_plot:
            plot_tree(self.model,
                      feature_names=self.features_,
                      filled=True,
                      proportion=proportion,
                      rounded=True)
            plt.show()
        else:
            output_path = "./output/MLModels/tree.dot"  # TODO choose name
            dotfile = open(output_path, 'w')
            export_graphviz(self.model, out_file=output_path,
                            feature_names=self.features_,
                            class_names=self.classes_,
                            rounded=True, proportion=proportion,
                            precision=3, filled=True)
            dotfile.close()
            cwd = os.getcwd()  # get current directory
            os.chdir("C:/Users/awindmann/AppData/Local/Continuum/anaconda3/Library/bin/graphviz")
            call(["./dot", "-Tpng", "C:/Users/awindmann/Documents/MA_local" + output_path[1:],
                  "-o", "C:/Users/awindmann/Documents/MA_local/output/MLModels/tree.png"])
            os.chdir(cwd)
            print("Tree plot successfully exported.")
