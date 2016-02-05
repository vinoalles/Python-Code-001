# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 09:42:17 2016

@author: vinodh
"""
from __future__ import division
import warnings
warnings.filterwarnings("ignore")
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

def model1(X_train, X_valid, y_train, y_valid):
    #Ridge Regression
    clf = linear_model.Ridge (alpha = 200)
    print(" ------ Start training Ridge Regressor")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    print(" ------ Finished training.")
    score = sqrt(mean_squared_error(y_valid, y_pred))
    print(" ------> Root Mean Square Error for model 1", score)
    return clf