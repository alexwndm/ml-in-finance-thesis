# functions for splitting the dataset into train and test sets. Includes Cross Validation

# general
import pandas as pd
import numpy as np

# ML
from sklearn.model_selection import KFold


# get a train test split ([train||test]).
def get_train_test_indices(dates, test_size=0.2, test_border=None, purge_window=120):
    dates_unique = dates.unique()
    if test_border is None:  # get rel test border
        # quantile for right train border
        train_border = dates_unique[np.quantile(range(1, len(dates_unique)), q=1 - test_size, interpolation="lower")]
        # train_border = pd.to_datetime(np.quantile(dates_unique.unique().astype("int64"), q=1 - test_size))
        # purged test border
        test_border = dates_unique[dates_unique >= train_border][purge_window]
    else:  # use test border
        train_border = dates_unique[dates_unique <= test_border][-purge_window]

    # convert dates_unique to idx
    train_border_idx = dates.get_loc(train_border)
    if type(train_border_idx) is not int and type(train_border_idx) is not np.int64:  # idx is slice
        train_border_idx = train_border_idx.stop  # last idx
    test_border_idx = dates.get_loc(test_border)
    if type(test_border_idx) is not int and type(test_border_idx) is not np.int64:
        test_border_idx = test_border_idx.start  # first idx

    train_indices = slice(0, train_border_idx)
    test_indices = slice(test_border_idx, dates.shape[0])

    return train_indices, test_indices


# for every fold (nb specified by cv), select train and test dates_unique.
# Purging and embargo included (See Lopez, p. 103 et seq.)
def get_cv_dates(dates, cv=5, purge_window=10, embargo_percentage=0.01):
    if cv < 2:
        raise ValueError("Cross Validation has to have at least 2 folds.")
    dates_unique = dates.unique()
    # select cv batches of same duration
    # train_test_borders = pd.to_datetime(np.quantile(dates_unique.unique().astype("int64"), q=[i / cv for i in range(1, cv)]))
    train_test_borders = dates_unique[np.quantile(range(0, len(dates_unique)), q=[i / cv for i in range(0, cv+1)],
                                                  interpolation="lower")]

    # train_test_borders_idx = [0]
    # for tr_tst_border in train_test_borders[1:]:
    #     tr_tst_border_idx = dates_unique.get_loc(tr_tst_border)
    #     if type(tr_tst_border_idx) is not int:  # idx is slice
    #         tr_tst_border_idx = tr_tst_border_idx.stop  # last idx
    #     train_test_borders_idx.append(tr_tst_border_idx)
    # batches_idx = [slice(train_test_borders_idx[i] , train_test_borders_idx[i + 1])
    #                for i in range(0, len(train_test_borders) - 1)]
    # TOD0 add last day

    batches = [dates_unique[(dates_unique > train_test_borders[fold])
                            & (dates_unique <= train_test_borders[fold + 1])]
               for fold in range(0, cv)]
    batches[0] = batches[0].append(dates[:1]).sort_values()  # add first day

    # combine batches to get train and test indices for every fold
    cv_dates = []
    for train_batches, test_batch in KFold(cv).split(range(cv)):
        # get batches relative to test_batch
        test_batch = int(test_batch)  # test batch
        train_batches_1 = train_batches[train_batches < test_batch]  # train batches before testing
        train_batches_2 = train_batches[train_batches > test_batch]  # train batches after testing

        # handle test batch first
        # get dates and purge if necessary
        test_dates = batches[test_batch]
        if test_batch > 0:  # if test not at start
            test_dates = test_dates[purge_window:]

        # convert dates to indices
        test_idx_start = dates.get_loc(test_dates[0])
        if type(test_idx_start) is not np.int64 and type(test_idx_start) is not int:  # idx is slice
            test_idx_start = test_idx_start.start  # first idx
        test_idx_stop = dates.get_loc(test_dates[-1])
        if type(test_idx_stop) is not np.int64 and type(test_idx_stop) is not int:  # idx is slice
            test_idx_stop = test_idx_stop.stop  # last idx
        test_indices = slice(test_idx_start, test_idx_stop + 1)

        # now combine train batches
        if len(train_batches_1) > 0:  # if there are train batches before test batch
            train1_idx_stop = dates.get_loc(batches[train_batches_1[-1]][-1])
            if type(train1_idx_stop) is not np.int64 and type(test_idx_stop) is not int:  # idx is slice
                train1_idx_stop = train1_idx_stop.stop  # last idx
            train_indices = slice(0, train1_idx_stop + 1)
            # train_dates = pd.DatetimeIndex.append(batches[train_batches_1[0]],
            #                                       [batches[i] for i in train_batches_1[1:]])
        if len(train_batches_2) > 0:  # if there are train batches after the test batch
            # purge train batch after test batch, add embargo
            train_dates_2 = pd.DatetimeIndex.append(
                batches[train_batches_2[0]][(purge_window + int(dates_unique.unique().shape[0] * embargo_percentage)):],
                [batches[i] for i in train_batches_2[1:]])

            # convert dates to indices
            train2_idx_start = dates.get_loc(train_dates_2[0])
            if type(train2_idx_start) is not np.int64 and type(test_idx_stop) is not int:  # idx is slice
                train2_idx_start = train2_idx_start.start  # first idx
            train2_indices = slice(train2_idx_start, dates.shape[0] + 1)

            if len(train_batches_1) == 0:
                train_indices = train2_indices
            else:  # combine batches
                train1_indices = np.arange(train_indices.start, train_indices.stop)
                train2_indices = np.arange(train2_indices.start, train2_indices.stop - 1)
                train_indices = np.concatenate([train1_indices, train2_indices])

        cv_dates.append({"train_dates": train_indices, "test_dates": test_indices})
    return cv_dates

