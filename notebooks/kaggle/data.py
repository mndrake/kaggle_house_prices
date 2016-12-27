# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder

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

create_dummies=True
def load_data_adjustable(create_dummies=False,
                         variance_threshold=0.,
                         drop_columns=None):
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
    if drop_columns is not None:
        combined.drop(drop_columns, axis=1, inplace=True)
    for col in combined.columns:
        if combined[col].dtype == 'object':
            combined.loc[combined[col].isnull(), col] = 'NA'
            if not create_dummies:
                combined[col] = pd.factorize(combined[col])[0]
        else:
            col_mean = combined[col].mean()
            combined[col] = combined[col].fillna(col_mean).values - col_mean

    if create_dummies:
        combined = pd.get_dummies(combined)

    combined = combined.values

    # drop low variance columns
    selector = VarianceThreshold(variance_threshold)
    combined = selector.fit_transform(combined)

    X = combined[:train.shape[0], :]
    X_submission = combined[train.shape[0]:, :]

    ids_submission = test['Id'].values
    y = train['SalePrice'].values

    return y, X, X_submission, ids_submission