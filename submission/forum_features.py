import numpy as np
import pandas as pd
from scipy.stats import boxcox
#import warnings
#from sklearn.linear_model import LassoCV, Lasso, ElasticNet
from sklearn.svm import SVC
from itertools import chain, product
#from sklearn.ensemble import BaggingRegressor
#from sklearn.model_selection import cross_val_score, KFold
#from sklearn.metrics import mean_squared_error
#import matplotlib.pyplot as plt
#import xgboost as xgb


def load_data():
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    combined = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'], test.loc[:, 'MSSubClass':'SaleCondition']), ignore_index=True)

    # warnings.simplefilter('ignore', np.RankWarning)

    # *** Missing Data ***

    # interpolate LotFrontage from LotArea
    keep_idx = ~(combined["LotFrontage"].isnull() | (combined["LotFrontage"] > 150) | (combined["LotArea"] > 25000))

    # polynomial coefficients
    p = np.polyfit(x=combined.ix[keep_idx, "LotArea"], y=combined.ix[keep_idx, "LotFrontage"], deg=1)
    combined.loc[combined['LotFrontage'].isnull(), 'LotFrontage'] = np.polyval(p, combined.loc[combined['LotFrontage'].isnull(), 'LotArea'])

    combined.fillna({
        'Alley': 'NoAlley',
        'MasVnrType': 'None',
        'FireplaceQu': 'NoFireplace',
        'GarageType': 'NoGarage',
        'GarageFinish': 'NoGarage',
        'GarageQual': 'NoGarage',
        'GarageCond': 'NoGarage',
        'BsmtFullBath': 0,
        'BsmtHalfBath': 0,
        'BsmtQual': 'NoBsmt',
        'BsmtCond': 'NoBsmt',
        'BsmtExposure': 'NoBsmt',
        'BsmtFinType1': 'NoBsmt',
        'BsmtFinType2': 'NoBsmt',
        'KitchenQual': 'TA',
        'MSZoning': 'RL',
        'Utilities': 'AllPub',
        'Exterior1st': 'VinylSd',
        'Exterior2nd': 'VinylSd',
        'Functional': 'Typ',
        'PoolQC': 'NoPool',
        'Fence': 'NoFence',
        'MiscFeature': 'None',
        'Electrical': 'SBrkr',
        'MasVnrArea': 0,
    }, inplace=True)

    combined.loc[combined['BsmtFinType1'] == 'NoBsmt', 'BsmtFinSF1'] = 0
    combined.loc[combined['BsmtFinType2'] == 'NoBsmt', 'BsmtFinSF2'] = 0
    combined.loc[combined['BsmtFinSF1'].isnull(), 'BsmtFinSF1'] = combined.BsmtFinSF1.median()
    combined.loc[combined['BsmtQual'] == 'NoBsmt', 'BsmtUnfSF'] = 0
    combined.loc[combined['BsmtUnfSF'].isnull(), 'BsmtUnfSF'] = combined.BsmtUnfSF.median()
    combined.loc[combined['BsmtQual'] == 'NoBsmt', 'TotalBsmtSF'] = 0

    # only one is null and it has type Detchd
    combined.loc[combined['GarageArea'].isnull(), 'GarageArea'] = combined.loc[combined['GarageType'] == 'Detchd', 'GarageArea'].mean()
    combined.loc[combined['GarageCars'].isnull(), 'GarageCars'] = combined.loc[combined['GarageType'] == 'Detchd', 'GarageCars'].median()

    # where we have order we will use numeric
    combined = combined.replace({'Utilities': {'AllPub': 1, 'NoSeWa': 0, 'NoSewr': 0, 'ELO': 0},
                                 'Street': {'Pave': 1, 'Grvl': 0 },
                                 'FireplaceQu': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NoFireplace': 0},
                                 'Fence': {'GdPrv': 2, 'GdWo': 2, 'MnPrv': 1, 'MnWw': 1, 'NoFence': 0},
                                 'ExterQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
                                 'ExterCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
                                 'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NoBsmt': 0},
                                 'BsmtExposure': {'Gd': 3, 'Av': 2, 'Mn': 1, 'No': 0, 'NoBsmt': 0},
                                 'BsmtCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NoBsmt': 0},
                                 'GarageQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NoGarage': 0},
                                 'GarageCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NoGarage': 0},
                                 'KitchenQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
                                 'Functional': {'Typ': 0, 'Min1': 1, 'Min2': 1, 'Mod': 2, 'Maj1': 3, 'Maj2': 4, 'Sev': 5, 'Sal': 6}
                                 })

    combined = combined.replace({'CentralAir': {'Y': 1, 'N': 0}})
    combined = combined.replace({'PavedDrive': {'Y': 1, 'P': 0, 'N': 0}})


    # ** Additional Features **

    newer_dwelling = combined.MSSubClass.replace({20: 1, 30: 0, 40: 0, 45: 0, 50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,
                                                  90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})
    newer_dwelling.name = 'newer_dwelling'

    combined = combined.replace({'MSSubClass': {20: 'SubClass_20', 30: 'SubClass_30', 40: 'SubClass_40',
                                                45: 'SubClass_45', 50: 'SubClass_50', 60: 'SubClass_60',
                                                70: 'SubClass_70', 75: 'SubClass_75', 80: 'SubClass_80',
                                                85: 'SubClass_85', 90: 'SubClass_90', 120: 'SubClass_120',
                                                150: 'SubClass_150', 160: 'SubClass_160', 180: 'SubClass_180',
                                                190: 'SubClass_190'}})

    # The idea is good quality should rise price, poor quality - reduce price
    overall_poor_qu = combined.OverallQual.copy()
    overall_poor_qu = 5 - overall_poor_qu
    overall_poor_qu[overall_poor_qu < 0] = 0
    overall_poor_qu.name = 'overall_poor_qu'

    overall_good_qu = combined.OverallQual.copy()
    overall_good_qu = overall_good_qu - 5
    overall_good_qu[overall_good_qu < 0] = 0
    overall_good_qu.name = 'overall_good_qu'

    overall_poor_cond = combined.OverallCond.copy()
    overall_poor_cond = 5 - overall_poor_cond
    overall_poor_cond[overall_poor_cond < 0] = 0
    overall_poor_cond.name = 'overall_poor_cond'

    overall_good_cond = combined.OverallCond.copy()
    overall_good_cond = overall_good_cond - 5
    overall_good_cond[overall_good_cond < 0] = 0
    overall_good_cond.name = 'overall_good_cond'

    exter_poor_qu = combined.ExterQual.copy()
    exter_poor_qu[exter_poor_qu < 3] = 1
    exter_poor_qu[exter_poor_qu >= 3] = 0
    exter_poor_qu.name = 'exter_poor_qu'

    exter_good_qu = combined.ExterQual.copy()
    exter_good_qu[exter_good_qu <= 3] = 0
    exter_good_qu[exter_good_qu > 3] = 1
    exter_good_qu.name = 'exter_good_qu'

    exter_poor_cond = combined.ExterCond.copy()
    exter_poor_cond[exter_poor_cond < 3] = 1
    exter_poor_cond[exter_poor_cond >= 3] = 0
    exter_poor_cond.name = 'exter_poor_cond'

    exter_good_cond = combined.ExterCond.copy()
    exter_good_cond[exter_good_cond <= 3] = 0
    exter_good_cond[exter_good_cond > 3] = 1
    exter_good_cond.name = 'exter_good_cond'

    bsmt_poor_cond = combined.BsmtCond.copy()
    bsmt_poor_cond[bsmt_poor_cond < 3] = 1
    bsmt_poor_cond[bsmt_poor_cond >= 3] = 0
    bsmt_poor_cond.name = 'bsmt_poor_cond'

    bsmt_good_cond = combined.BsmtCond.copy()
    bsmt_good_cond[bsmt_good_cond <= 3] = 0
    bsmt_good_cond[bsmt_good_cond > 3] = 1
    bsmt_good_cond.name = 'bsmt_good_cond'

    garage_poor_qu = combined.GarageQual.copy()
    garage_poor_qu[garage_poor_qu < 3] = 1
    garage_poor_qu[garage_poor_qu >= 3] = 0
    garage_poor_qu.name = 'garage_poor_qu'

    garage_good_qu = combined.GarageQual.copy()
    garage_good_qu[garage_good_qu <= 3] = 0
    garage_good_qu[garage_good_qu > 3] = 1
    garage_good_qu.name = 'garage_good_qu'

    garage_poor_cond = combined.GarageCond.copy()
    garage_poor_cond[garage_poor_cond < 3] = 1
    garage_poor_cond[garage_poor_cond >= 3] = 0
    garage_poor_cond.name = 'garage_poor_cond'

    garage_good_cond = combined.GarageCond.copy()
    garage_good_cond[garage_good_cond <= 3] = 0
    garage_good_cond[garage_good_cond > 3] = 1
    garage_good_cond.name = 'garage_good_cond'

    kitchen_poor_qu = combined.KitchenQual.copy()
    kitchen_poor_qu[kitchen_poor_qu < 3] = 1
    kitchen_poor_qu[kitchen_poor_qu >= 3] = 0
    kitchen_poor_qu.name = 'kitchen_poor_qu'

    kitchen_good_qu = combined.KitchenQual.copy()
    kitchen_good_qu[kitchen_good_qu <= 3] = 0
    kitchen_good_qu[kitchen_good_qu > 3] = 1
    kitchen_good_qu.name = 'kitchen_good_qu'

    qu_list = pd.concat((overall_poor_qu, overall_good_qu, overall_poor_cond, overall_good_cond, exter_poor_qu,
                         exter_good_qu, exter_poor_cond, exter_good_cond, bsmt_poor_cond, bsmt_good_cond,
                         garage_poor_qu,
                         garage_good_qu, garage_poor_cond, garage_good_cond, kitchen_poor_qu, kitchen_good_qu), axis=1)

    bad_heating = combined.HeatingQC.replace({'Ex': 0, 'Gd': 0, 'TA': 0, 'Fa': 1, 'Po': 1})
    bad_heating.name = 'bad_heating'

    MasVnrType_Any = combined.MasVnrType.replace({'BrkCmn': 1, 'BrkFace': 1, 'CBlock': 1, 'Stone': 1, 'None': 0})
    MasVnrType_Any.name = 'MasVnrType_Any'

    SaleCondition_PriceDown = combined.SaleCondition.replace({'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1,
                                                              'Normal': 0, 'Partial': 0})
    SaleCondition_PriceDown.name = 'SaleCondition_PriceDown'

    Neighborhood_Good = pd.DataFrame(np.zeros((combined.shape[0], 1)), columns=['Neighborhood_Good'])
    Neighborhood_Good[combined.Neighborhood == 'NridgHt'] = 1
    Neighborhood_Good[combined.Neighborhood == 'Crawfor'] = 1
    Neighborhood_Good[combined.Neighborhood == 'StoneBr'] = 1
    Neighborhood_Good[combined.Neighborhood == 'Somerst'] = 1
    Neighborhood_Good[combined.Neighborhood == 'NoRidge'] = 1

    svm = SVC(C=100, gamma=0.0001, kernel='rbf')
    pc = pd.Series(np.zeros(train.shape[0]))

    pc[:] = 'pc1'
    pc[train.SalePrice >= 150000] = 'pc2'
    pc[train.SalePrice >= 220000] = 'pc3'
    columns_for_pc = ['Exterior1st', 'Exterior2nd', 'RoofMatl', 'Condition1', 'Condition2', 'BldgType']
    X_t = pd.get_dummies(train.loc[:, columns_for_pc], sparse=True)
    svm.fit(X_t, pc)  # Training
    pc_pred = svm.predict(X_t)

    p = train.SalePrice / 100000

    price_category = pd.DataFrame(np.zeros((combined.shape[0], 1)), columns=['pc'])
    X_t = pd.get_dummies(combined.loc[:, columns_for_pc], sparse=True)
    pc_pred = svm.predict(X_t)
    price_category[pc_pred == 'pc2'] = 1
    price_category[pc_pred == 'pc3'] = 2

    price_category = price_category.to_sparse()

    season = combined.MoSold.replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})
    season.name = 'season'

    combined = combined.replace({'MoSold': {1: 'Yan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul',
                                            8: 'Avg', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}})

    reconstruct = pd.DataFrame(np.zeros((combined.shape[0], 1)), columns=['Reconstruct'])
    reconstruct[combined.YrSold < combined.YearRemodAdd] = 1
    reconstruct = reconstruct.to_sparse()

    recon_after_buy = pd.DataFrame(np.zeros((combined.shape[0], 1)), columns=['ReconstructAfterBuy'])
    recon_after_buy[combined.YearRemodAdd >= combined.YrSold] = 1
    recon_after_buy = recon_after_buy.to_sparse()

    build_eq_buy = pd.DataFrame(np.zeros((combined.shape[0], 1)), columns=['Build.eq.Buy'])
    build_eq_buy[combined.YearBuilt >= combined.YrSold] = 1
    build_eq_buy = build_eq_buy.to_sparse()

    combined.YrSold = 2010 - combined.YrSold
    year_map = pd.concat(
        pd.Series('YearGroup' + str(i + 1), index=range(1871 + i * 20, 1891 + i * 20)) for i in range(0, 7))
    combined.GarageYrBlt = combined.GarageYrBlt.map(year_map)
    combined.loc[combined['GarageYrBlt'].isnull(), 'GarageYrBlt'] = 'NoGarage'

    combined.YearBuilt = combined.YearBuilt.map(year_map)
    combined.YearRemodAdd = combined.YearRemodAdd.map(year_map)

    # ** Scale Features **

    numeric_feats = combined.dtypes[combined.dtypes != "object"].index
    t = combined[numeric_feats].quantile(.75)
    use_75_scater = t[t != 0].index
    combined[use_75_scater] = combined[use_75_scater] / combined[use_75_scater].quantile(.75)

    t = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
         '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
         'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
    
    combined.loc[:, t] = np.log1p(combined.loc[:, t])
    
    combined['GrLivArea'], _ = boxcox(combined['GrLivArea']) 
    
    #train["SalePrice"] = np.log1p(train["SalePrice"])
    train["SalePrice"] = np.log(train["SalePrice"])

    X = pd.get_dummies(combined)
    X = X.fillna(X.mean())

    X = X.drop('RoofMatl_ClyTile', axis=1)  # only one is not zero
    X = X.drop('Condition2_PosN', axis=1)  # only two is not zero
    X = X.drop('MSZoning_C (all)', axis=1)
    X = X.drop('MSSubClass_SubClass_160', axis=1)

    X = pd.concat((X, newer_dwelling, season, reconstruct, recon_after_buy,
                   qu_list, bad_heating, MasVnrType_Any, price_category, build_eq_buy), axis=1)

    def poly(X):
        areas = ['LotArea', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'BsmtUnfSF']
        t = chain(qu_list.axes[1].get_values(),
                  ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtCond', 'GarageQual', 'GarageCond',
                   'KitchenQual', 'HeatingQC', 'bad_heating', 'MasVnrType_Any', 'SaleCondition_PriceDown',
                   'Reconstruct',
                   'ReconstructAfterBuy', 'Build.eq.Buy'])
        for a, t in product(areas, t):
            x = X.loc[:, [a, t]].prod(1)
            x.name = a + '_' + t
            yield x

    XP = pd.concat(poly(X), axis=1)
    X = pd.concat((X, XP), axis=1)

    X_train = X[:train.shape[0]]
    X_test = X[train.shape[0]:]
    y = train.SalePrice

    outliers_id = np.array([523, 1298])

    X_train = X_train.drop(outliers_id)
    y = y.drop(outliers_id)

    ids_submission = test['Id'].values

    return X_train, y, X_test, ids_submission
