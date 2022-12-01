#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 19:51:11 2022

@author: pranay
"""

import numpy as np 
import sklearn as sk 
import pandas as pd
import sklearn.linear_model

data_dir = '/Users/pranay/Desktop/MATH 373/homework/homework3/'

fname = 'train.csv'

df_train = pd.read_csv(data_dir + fname)
#columns=['Sex','Age']
columns=['Sex','Age','Embarked','Pclass','Fare']
#'Pclass','SibSp','Fare','Parch'

X_train = df_train[columns]

mean_age = np.mean(X_train['Age'])
X_train['Age'].fillna(mean_age, inplace = True)

mean_fare = np.mean(X_train['Fare'])
X_train['Fare'].fillna(mean_fare,inplace=True)

mode_embarked= X_train['Embarked'].mode()
#print(mode_embarked)
X_train['Embarked'].fillna(mode_embarked.iloc[0], inplace=True)

one_hot_sex = pd.get_dummies(X_train['Sex'])
X_train = X_train.drop(columns = ['Sex'])
X_train = X_train.join(one_hot_sex)


one_hot_embarked = pd.get_dummies(X_train['Embarked'])
X_train = X_train.drop(columns=['Embarked'])
X_train = X_train.join(one_hot_embarked)


one_hot_Pclass = pd.get_dummies(X_train['Pclass'])
X_train = X_train.drop(columns=['Pclass'])
X_train = X_train.join(one_hot_Pclass)


#X_train = X_train.to_numpy()
#X_train = np.float64(X_train)


y_train = df_train['Survived'].to_numpy()

model = sk.linear_model.LogisticRegression()
model.fit(X_train,y_train)

fname = 'test.csv'
df_test = pd.read_csv(data_dir + fname)
X_test = df_test[columns]

one_hot_sex_test = pd.get_dummies(X_test['Sex'])
X_test = X_test.drop(columns = ['Sex'])
X_test = X_test.join(one_hot_sex_test)

one_hot_embarked_test = pd.get_dummies(X_test['Embarked'])
X_test = X_test.drop(columns=['Embarked'])
X_test = X_test.join(one_hot_embarked_test)

one_hot_Pclass_test = pd.get_dummies(X_test['Pclass'])
X_test = X_test.drop(columns=['Pclass'])
X_test = X_test.join(one_hot_Pclass_test)


X_test['Age'].fillna(mean_age, inplace = True)

X_test['Fare'].fillna(mean_fare,inplace = True)

X_test = X_test.to_numpy()
X_test = np.float64(X_test)


y_pred = model.predict(X_test)

kaggle_submission = df_test['PassengerId'].to_frame()
kaggle_submission['Survived'] = y_pred

kaggle_submission.to_csv(data_dir + 'myKaggleSubmission.csv', index = False)

