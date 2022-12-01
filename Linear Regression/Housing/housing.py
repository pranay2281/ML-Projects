#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import sklearn as sk
import sklearn.datasets 
import pandas as pd
import sklearn.linear_model

# this is question 1 of the homework 
data_dir = '/Users/pranay/Desktop/MATH 373/homework/homework2/'


fname = 'train.csv'


df_train = pd.read_csv(data_dir + fname)
columns = ['LotArea','LotFrontage', 'OverallQual', 'OverallCond','YearBuilt', 'YearRemodAdd', 
           'Fireplaces', 'GarageArea', 'CentralAir', 'YrSold','Foundation', 'RoofStyle',
           'GarageCond','SaleCondition','PoolArea'] 

# ExterQual
# TotalBsmtSF



df_train = pd.read_csv(data_dir + fname)
X_train = df_train[columns]
# X_train.fillna(0, inplace = True)


one_hot = pd.get_dummies(X_train['CentralAir'])
X_train = X_train.drop(columns = ['CentralAir'])
X_train = X_train.join(one_hot)

one_hot_foundation = pd.get_dummies(X_train['Foundation'])
X_train = X_train.drop(columns = ['Foundation'])
X_train = X_train.join(one_hot_foundation)

one_hot_roofstyle = pd.get_dummies(X_train['RoofStyle'])
X_train = X_train.drop(columns = ['RoofStyle'])
X_train = X_train.join(one_hot_roofstyle)


one_hot_garagecond = pd.get_dummies(X_train['GarageCond'])
X_train = X_train.drop(columns=['GarageCond'])
X_train = X_train.join(one_hot_garagecond)

one_hot_salecondition = pd.get_dummies(X_train['SaleCondition'])
X_train = X_train.drop(columns=['SaleCondition'])
X_train = X_train.join(one_hot_salecondition)



mean_lot_front = np.mean(X_train['LotFrontage'])
X_train['LotFrontage'].fillna(mean_lot_front, inplace = True)

mean_garage_area = np.mean(X_train['GarageArea'])
X_train['GarageArea'].fillna(mean_garage_area, inplace = True)


X_train = X_train.to_numpy()
X_train = np.float64(X_train)

y_train = df_train['SalePrice'].to_numpy()



model = sk.linear_model.LinearRegression()
model.fit(X_train, y_train)


fname = 'test.csv'
df_test = pd.read_csv(data_dir + fname)
X_test = df_test[columns] #data frame test on these columns

one_hot_test = pd.get_dummies(X_test['CentralAir'])
X_test = X_test.drop(columns = ['CentralAir'])
X_test = X_test.join(one_hot_test)

one_hot_foundation_test = pd.get_dummies(X_test['Foundation'])
X_test = X_test.drop(columns = ['Foundation'])
X_test = X_test.join(one_hot_foundation_test)

one_hot_roofstyle_test = pd.get_dummies(X_test['RoofStyle'])
X_test = X_test.drop(columns = ['RoofStyle'])
X_test = X_test.join(one_hot_roofstyle)

one_hot_garagecond_test = pd.get_dummies(X_test['GarageCond'])
X_test = X_test.drop(columns=['GarageCond'])
X_test = X_test.join(one_hot_garagecond_test)

one_hot_salecondition = pd.get_dummies(X_test['SaleCondition'])
X_test = X_test.drop(columns=['SaleCondition'])
X_test = X_test.join(one_hot_salecondition)


X_test['LotFrontage'].fillna(mean_lot_front, inplace = True)
X_test['GarageArea'].fillna(mean_garage_area, inplace = True)

X_test = X_test.to_numpy()
X_test = np.float64(X_test)


y_pred = model.predict(X_test)


kaggle_submission = df_test['Id'].to_frame()
kaggle_submission['SalePrice'] = y_pred

kaggle_submission.to_csv(data_dir + 'myKaggleSubmission.csv', index = False)


# 
