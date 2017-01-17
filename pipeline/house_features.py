import numpy as np
import pandas as pd
from sklearn.pipeline import BaseEstimator, TransformerMixin


class CategoricalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        
        X = X[['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'Alley', 'BedroomAbvGr', 'BldgType', 'BsmtCond',
               'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath',
               'BsmtHalfBath', 'BsmtQual', 'BsmtUnfSF', 'CentralAir', 'Condition1', 'Condition2',
               'Electrical', 'EnclosedPorch', 'ExterCond', 'ExterQual', 'Exterior1st', 'Exterior2nd',
               'Fence', 'FireplaceQu', 'Fireplaces', 'Foundation', 'FullBath', 'Functional',
               'GarageArea', 'GarageCars', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType',
               'GarageYrBlt', 'GrLivArea', 'HalfBath', 'Heating', 'HeatingQC', 'HouseStyle',
               'KitchenAbvGr', 'KitchenQual', 'LandContour', 'LandSlope', 'LotArea', 'LotConfig',
               'LotFrontage', 'LotShape', 'LowQualFinSF', 'MSSubClass', 'MSZoning', 'MasVnrArea',
               'MasVnrType', 'MiscFeature', 'MiscVal', 'MoSold', 'Neighborhood', 'OpenPorchSF',
               'OverallCond', 'OverallQual', 'PavedDrive', 'PoolArea', 'PoolQC', 'RoofMatl',
               'RoofStyle', 'SaleCondition', 'SaleType', 'ScreenPorch', 'Street', 'TotRmsAbvGrd',
               'TotalBsmtSF', 'Utilities', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'YrSold']]
        
        # MSSubClass is a categorical and need to cast to object
        X["MSSubClass"] = X["MSSubClass"].astype('object')
        
        categorical = {
            "ordered": {
                "Alley": ["Grvl", "Pave"],
                "BsmtCond": ["Po", "Fa", "TA", "Gd"],
                "BsmtExposure": ["No", "Mn", "Av", "Gd"],
                "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
                "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
                "BsmtQual": ["Fa", "TA", "Gd", "Ex"],
                "CentralAir": ["N", "Y"],
                "Electrical": ["FuseP", "FuseF", "FuseA", "Mix", "SBrkr"],
                "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
                "ExterQual": ["Fa", "TA", "Gd", "Ex"],
                "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
                "FireplaceQu": ["Po", "Fa", "TA", "Gd", "Ex"],
                'Functional': ['Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
                "GarageCond": ["Po", "Fa", "TA", "Gd", "Ex"],
                "GarageFinish": ["Unf", "RFn", "Fin"],
                "GarageQual": ["Po", "Fa", "TA", "Gd", "Ex"],
                "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
                "KitchenQual": ["Fa", "TA", "Gd", "Ex"],
                "LandSlope": ["Sev", "Mod", "Gtl"],
                "LotShape": ["IR3", "IR2", "IR1", "Reg"],
                "PavedDrive": ["N", "P", "Y"],
                "PoolQC": ["Fa", "Gd", "Ex"],
                "Street": ["Grvl", "Pave"],   
                "Utilities": ["NoSeWa", "AllPub"]},
            "unordered": {
                "BldgType": ["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"],
                "Exterior1st": ["VinylSd", "MetalSd", "Wd Sdng", "HdBoard", "BrkFace", "WdShing", "CemntBd", "Plywood", "AsbShng", "Stucco", "BrkComm", "AsphShn", "Stone", "ImStucc", "CBlock"],
                "Exterior2nd": ["VinylSd", "MetalSd", "Wd Shng", "HdBoard", "Plywood", "Wd Sdng", "CmentBd", "BrkFace", "Stucco", "AsbShng", "Brk Cmn", "ImStucc", "AsphShn", "Stone", "Other", "CBlock"],
                "Condition1": ["Norm", "Feedr", "PosN", "Artery", "RRAe", "RRNn", "RRAn", "PosA", "RRNe"],
                "Condition2": ["Norm", "Artery", "RRNn", "Feedr", "PosN", "PosA", "RRAn", "RRAe"],
                "Foundation": ["PConc", "CBlock", "BrkTil", "Wood", "Slab", "Stone"],
                "GarageType": ["Attchd", "Detchd", "BuiltIn", "CarPort", "Basment", "2Types"],
                "Heating": ["GasA", "GasW", "Grav", "Wall", "OthW", "Floor"],
                "HouseStyle": ["2Story", "1Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl", "2.5Unf", "2.5Fin"],
                "LandContour": ["Lvl", "Bnk", "Low", "HLS"],
                "LotConfig": ["Inside", "FR2", "Corner", "CulDSac", "FR3"],
                "MasVnrType": ["BrkFace", "None", "Stone", "BrkCmn"],
                "MSSubClass": [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190],
                "MSZoning": ["RL", "RM", "C (all)", "FV", "RH"],
                "Neighborhood": ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", "NWAmes", "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes", "SawyerW", "IDOTRR", "MeadowV", "Edwards", "Timber", "Gilbert", "StoneBr", "ClearCr", "NPkVill", "Blmngtn", "BrDale", "SWISU", "Blueste"],
                "MiscFeature": ["Shed", "Gar2", "Othr", "TenC"],
                "RoofMatl": ["CompShg", "WdShngl", "Metal", "WdShake", "Membran", "Tar&Grv", "Roll", "ClyTile"],
                "RoofStyle": ["Gable", "Hip", "Gambrel", "Mansard", "Flat", "Shed"],
                "SaleCondition": ["Normal", "Abnorml", "Partial", "AdjLand", "Alloca", "Family"],
                "SaleType": ["WD", "New", "COD", "ConLD", "ConLI", "CWD", "ConLw", "Con", "Oth"]}}
        
        for c in X.columns:
            if c in categorical["ordered"]:
                X[c] = X[c].astype("category", categories=categorical["ordered"][c], ordered=True)
            elif c in categorical["unordered"]:
                X[c] = X[c].astype("category", categories=categorical["unordered"][c])
                
        return X
        
        
class TreeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        X["HasFireplace"] = 1 - X["FireplaceQu"].isnull() * 1
        X["AttchdGarage"] = (X['GarageType'] == "Attchd") * 1
                
        for c in X.columns:
            if X[c].dtype.name == 'category':
                if X[c].cat.ordered:
                    X[c] = X[c].cat.codes
                    
        return pd.get_dummies(X)
        
        
class LinearFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        X["HasFireplace"] = 1 - X["FireplaceQu"].isnull() * 1
        X["AttchdGarage"] = (X['GarageType'] == "Attchd") * 1
        
        for c in X.columns:
            if X[c].dtype.name == 'category':
                if X[c].cat.ordered:
                    X[c] = X[c].cat.codes
                    
        # skewed columns
        for c in ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF2', '1stFlrSF', 
                  'GrLivArea', 'KitchenAbvGr', 'OpenPorchSF', 'PoolArea', 'MiscVal']:
            X[c] = np.log1p(X[c])
                                            
        return pd.get_dummies(X, drop_first=True)