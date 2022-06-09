# Asset allocators

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

from functions.HRP_functions import plotCorrMatrix, correlDist, getQuasiDiag, getRecBipart


# equal weight asset allocator
class EWAllocator:
    def __init__(self, max_weight=0.05):
        self.method = "EqualWeight"
        self.max_weight = max_weight

    # allocate the assets in the portfolio. return portfolio weights in the form: (cash, assets).
    def get_weights(self, curr_df, portfolio_symbols):
        if portfolio_symbols.shape[0] == 0:  # empty portfolio
            weights = np.array([1])  # all in cash
        elif portfolio_symbols.shape[0] <= 1 / self.max_weight:  # small portfolio
            n = len(portfolio_symbols)
            weights = np.repeat(self.max_weight, n + 1)
            weights[0] = 1 - n * self.max_weight  # rest in cash
        else:
            n = len(portfolio_symbols)
            weights = np.repeat(1 / n, n + 1)
            weights[0] = 0  # no cash
        return weights


# Hierarchical Risk Parity
class HRPAllocator:
    def __init__(self, ret_mat, max_weight=1, ret_window=5):
        self.method = "HRP"
        self.ret_mat = ret_mat
        self.max_weight = max_weight
        self.ret_window = ret_window

    # for details, see Advances in Financial Machine Learning, Lopez de Prado, 2018, pp.221-231
    def get_weights(self, curr_df, portfolio_symbols):
        if portfolio_symbols.shape[0] == 0:  # empty portfolio
            weights = np.array([1])  # all in cash
        elif portfolio_symbols.shape[0] == 1:  # no correlation structure for single asset
            weights = np.array([1 - self.max_weight, self.max_weight])  # (cash, asset)
        else:
            # get current day
            curr_day = curr_df.index[0]

            # get correlation structure
            curr_ret_mat = self.ret_mat.loc[self.ret_mat.index <= curr_day, portfolio_symbols][-252*self.ret_window:] #TODO
            corr = curr_ret_mat.corr().fillna(0)
            cov = curr_ret_mat.cov().fillna(0)
            # corr.values[[np.arange(corr.shape[0])] * 2] = 1  # diag values can be replaced by 0 as well  TODO
            # plotCorrMatrix("output/HRP_corr0.png", corr, labels=corr.columns)

            # cluster assets
            dist = correlDist(corr)  # dist matrix
            link = sch.linkage(squareform(dist), 'single')
            sortIx = getQuasiDiag(link)
            sortIx = corr.index[sortIx].tolist()  # recover labels
            df0 = corr.loc[sortIx, sortIx]  # reorder
            # plotCorrMatrix("output/HRP_corr1.png", df0, labels=df0.columns)

            # capital allocation
            hrp = getRecBipart(cov, sortIx)
            weights = np.array([0])  # cash
            weights = np.append(weights, hrp[portfolio_symbols])
            for i in range(1, len(weights)):
                if weights[i] > self.max_weight:
                    weights[0] += weights[i] - self.max_weight  # stock up cash
                    weights[i] = self.max_weight  # limit weight
        return weights
