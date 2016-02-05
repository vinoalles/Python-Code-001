# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 09:26:49 2016

@author: vinodh
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

def load_train_data():
    print(" -- Start Reading training data")
    df_train = pd.read_csv('/Users/vinodh/Downloads/datatrain.csv')
    print(" -- End Reading training data")
    return df_train  
    

def load_test_data(path=None):
    print(" --- Start Reading testing data")
    df_test = pd.read_csv('/Users/vinodh/Downloads/datatest.csv')
    print(" --- End Reading testing data")
    return df_test
