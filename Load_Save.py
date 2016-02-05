# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:05:13 2016

@author: vinodh
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import xgboost as xgb
import State_Farm_Read
import State_Farm_Data_Prep
import State_Farm_Model1
import State_Farm_Model2

def make_submission(clf, X_test, id_test, model):
    y_prob = clf.predict(X_test)
    id_test = pd.DataFrame(id_test)
    y_prob = pd.DataFrame(y_prob)
    output = pd.concat([y_prob, id_test],axis=1)
    output = output.rename(columns = {0:'X1'})
    if (model == 1):
        output.to_csv('/Users/vinodh/Downloads/stfarm/Results_from_Vinodh_model1.csv')
    if (model == 2):
        output.to_csv('/Users/vinodh/Downloads/stfarm/Results_from_Vinodh_model2.csv')
        

def main():
    print(" - Start.")
    df_train = State_Farm_Read.load_train_data()
    df_test = State_Farm_Read.load_test_data()
    X_train, X_valid, y_train, y_valid, X_test, id_test = State_Farm_Data_Prep.prep_data(df_train,df_test)
    clf = State_Farm_Model1.model1(X_train, X_valid, y_train, y_valid)
    model = 1
    make_submission(clf, X_test, id_test, model)
    model = 2
    clf = State_Farm_Model2.model2(X_train, X_valid, y_train, y_valid)
    X_test  = xgb.DMatrix(X_test)
    make_submission(clf, X_test, id_test, model)
    print("output file created in this location /Users/vinodh/Downloads/stfarm/")
    print(" --------- Finished.")

if __name__ == '__main__':
    main()
    
    