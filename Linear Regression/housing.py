#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 21:47:01 2021

@author: pranay
"""

import numpy as np
import sklearn as sk
import sklearn.datasets
import sklearn.model_selection

import sklearn.linear_model

dataset = sk.datasets.fetch_california_housing()

feature_vectors = dataset.data
Y = dataset.target
column_names = dataset.feature_names


X_train, X_val, Y_train, Y_val = sk.model_selection.train_test_split(feature_vectors, Y, train_size=0.8, random_state = 123)
mu = np.mean(X_train, axis = 0)
sigma = np.std(X_train, axis = 0)
X_train = (X_train-mu)/sigma

X_val = (X_val-mu)/sigma

X_train= np.insert(X_train, 0, 1, axis =1)
X_val = np.insert(X_val, 0 , 1, axis =1)

d = X_train.shape[1]-1

def L(X, Beta, y):
    lb =  X @ Beta - y
    norm = np.linalg.norm(lb)
    N = X.shape[0]
    return (1/N)*((norm)**2)

def gradient(X, Beta_est, y):
    X_transpose = X.T
    lb = X @ Beta_est - y
    N = X.shape[0]
    return (2/N)*(X_transpose @ lb)

def descent(X, y, step_size, epsilon):
    
    Beta_est = np.zeros(d+1)
    count = 0
    
    while(count < 500):
        
        gradient_est = gradient(X, Beta_est, y)
        newBeta = Beta_est - (step_size*gradient_est)
        Beta_est = newBeta
        
        if(abs(L(X,Beta_est,y))<epsilon):
            break
        
        count+=1
          
    return Beta_est



if __name__== "__main__":
    
    Beta = np.zeros(d+1)
    step_size=10**-1
    epsilon = 10**-4
    
    des = descent(X_train,Y_train,step_size,epsilon)
    ldes = L(X_val,des,Y_val)
    print(ldes)
    
    
    

    
    
    



    
