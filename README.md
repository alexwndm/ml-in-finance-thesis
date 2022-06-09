# Machine Learning in Finance

The code to the Master's Thesis "[Machine Learning in Finance](http://zeus.instmath.rwth-aachen.de/~maier/publications/Windmann2020.pdf)" by Alexander Windmann, 2020.

The program contains the whole financial machine learning process from generating a dataset, modeling ML classifiers, hyperparameter tuning and backtesting. 

The program is structured as follows:

* main.py:

    Run this to generate a dataset. Parameters that specify this dataset can be set in the beginning of the main.py file. 
    Furthermore, machine learning classifiers can be build and tested on the generated dataset, including hyperparameter tuning. 
    Output is in the subfiles output/FeatureCorrelations and output/MLModels.
	
* backtest_run.py:
	
    Use the generated dataset from main.py (which already is in the correct format).
    For the backtest, a selection scheme and an allocation method have to be specified. 
    Output is in the subfile output/Backtests.

* post_processing_run.py:
    
    Uses the equity curves that have been generated with backtest_run.py.
    Applies a number of trend indicators to the equity curves. 
    Output is in the subfile output/Backtests/PostProcessing.

Folder structure:

* classes:
    
    Contains the various classes that have been programmed. 

* data:
    
    Would contain the data. In the thesis, data of the S&P 500 and of the STOXX Europe 600 have been used.

* functions:
    
    Functions grouped respective to their tasks. Most important is feature_engineering.py, which specifies which features to apply to the dataset. 

* output:
    
    Contains plots and datasets that have been created.


