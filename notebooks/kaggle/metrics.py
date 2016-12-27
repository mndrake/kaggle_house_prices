# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def rmse_log(y_true, y_pred):
    return rmse(np.log(y_true), np.log(y_pred))