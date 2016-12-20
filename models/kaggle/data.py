# -*- coding: utf-8 -*-
import pandas as pd


def load_data():
    """
    process train and test data
    :return:
    (y, X, X_submission, ids_submission)
    """
    # read in data
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    # combine train and test to ease transformations
    combined = train.append(test, ignore_index=True)
    combined.drop(['Id', 'SalePrice'], axis=1, inplace=True)

    for col in combined.columns:
        if combined[col].dtype == 'object':
            combined[col] = combined[col].factorize()[0]
        else:
            combined[col] = combined[col].fillna(-99)

    X = combined.values[:train.shape[0], :]
    X_submission = combined.values[train.shape[0]:, :]

    ids_submission = test['Id'].values
    y = train['SalePrice'].values

    return y, X, X_submission, ids_submission


def save_submission(ids, y_pred, name):
    """save submission"""
    submission = pd.DataFrame({'Id': ids, 'SalePrice': y_pred})
    submission.to_csv('../output/{}.csv'.format(name), index=False)
