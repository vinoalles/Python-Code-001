# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 09:45:49 2016

@author: vinodh
"""

from __future__ import division
import warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

def model2(X_train, X_valid, y_train, y_valid):
    #Extreme Gradient Boosting
    params = {'gamma': 0.6,
           'max_depth': 6,
           'eval_metric': 'rmse',
           'silent': 0,
           'min_child_weight': 6.0,
           'colsample_bytree': 0.5,
           'nthread': 6,
           'subsample': 0.5,
           'eta': 0.375,
           'objective': 'reg:linear'}
    T_train_xgb = xgb.DMatrix(X_train, y_train)
    T_valid_xgb = xgb.DMatrix(X_valid, y_valid)    
    watchlist = [(T_valid_xgb, 'eval'), (T_train_xgb, 'train')]
    print(" ------- Start training XGBOOST Regressor")
    clf = xgb.train(params, T_train_xgb, 81, evals=watchlist, early_stopping_rounds=1)
    print(" ------- End training XGBOOST Regressor")
    return clf