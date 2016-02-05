# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 09:29:29 2016

@author: vinodh
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder

def prep_data(df_train,df_test,test_size=0.2):
    print(" ---- Start data prep")
    df_train = df_train.dropna(subset=['X1'])
    df_train['X1'] = (df_train['X1'].replace( '[\%,)]','',regex=True).replace( '[(]','-',   regex=True ).astype(float))
    labels = df_train['X1'].values
    id_test = df_test['X2']
    piv_train = df_train.shape[0]
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    del df_all['X1'], df_all['X2'], df_all['X3'], df_all['X10'], df_all['X16'], df_all['X18']
    df_all['X23'] = df_all['X23'].map(lambda x: str(x)[:-3])
    df_all['X15'] = df_all['X15'].map(lambda x: str(x)[:-3])
    df_all['X4'] = (df_all['X4'].replace( '[\$,)]','', regex=True).replace( '[(]','-',   regex=True ).astype(float))
    df_all['X5'] = (df_all['X5'].replace( '[\$,)]','', regex=True).replace( '[(]','-',   regex=True ).astype(float))
    df_all['X6'] = (df_all['X6'].replace( '[\$,)]','', regex=True).replace( '[(]','-',   regex=True ).astype(float))
    df_all['X30'] = (df_all['X30'].replace( '[\%,)]','', regex=True).replace( '[(]','-',   regex=True ).astype(float))
    df_f = feature_engineering(df_all)
    vals = df_f.values
    X = vals[:piv_train]
    le = LabelEncoder()
    y = le.fit_transform(labels)   
    y = labels 
    X_test = vals[piv_train:]
    X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(X, y, test_size=0.2)
    print(" ---- end data prep")
    return X_train, X_valid, y_train, y_valid, X_test, id_test


def feature_engineering(df_all):
    print(" ----- Start Feature Engineering")
    ohe_feats = ['X7', 'X8', 'X9', 'X11', 'X12', 'X14', 'X17', 'X19', 'X20', 'X32' , 'X23', 'X15']
    for f in ohe_feats:
        df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
        df_all = df_all.drop([f], axis=1)
        df_all = pd.concat((df_all, df_all_dummy), axis=1)
    #scaling and centering    
    df_f = ((df_all - df_all.mean())/df_all.std())
    #Filling nan
    df_f = df_f.fillna(-1)
    print(" ----- End Feature Engineering")
    return df_f