#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 20:20:24 2022

@author: pranay
"""

#question 2

#Layer 1: 100 nodes , (5+1)=6 parameters ,+1 is because of biased term = 600
#layer 2: 100+1 parameters, 100 nodes in layer 2 = 10100
#layer 3: 100+1 parameters, 100 nodes in layer 3 = 10100
#layer 4: 100+1 parameters, 100 nodes in layer 4 = 10100
#output layer = 3(classes) and 101 parameters = 303
# total = 600+10100+10100+10100+303 = 31203 total parameters




#import numpy as np
import sklearn as sk
import sklearn.datasets
import sklearn.model_selection
#import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.neural_network

dataset = sk.datasets.fetch_openml('mnist_784')
X = dataset.data
y = dataset.target
#print("Step 1")
X = X/255.0
#print("Step 2")

X_train, X_val, y_train,y_val = sk.model_selection.train_test_split(X,y)
#print("Step 3")

model_regression = sk.linear_model.LogisticRegression()
model_regression.fit(X_train,y_train)
#print("Step 4")
y_pred_regression = model_regression.predict(X_val)
#print("Step 5")
count_regression = 0

for j in range(len(y_val)):
        if(y_val[j]==y_pred_regression[j]):
            count_regression+=1

#print("Step 6")
            
nodes = [100,200]


count_mlp = []

for i in range(len(nodes)):
    n = nodes[i]
    model_mlp = sk.neural_network.MLPClassifier((n,n))
    model_mlp.fit(X_train,y_train)
    y_pred_mlp = model_mlp.predict(X_val)
    count=0
    for j in range(len(y_val)):
        if(y_val[j]==y_pred_mlp[j]):
            count+=1
    count_mlp.append(count)
    
    #print("Step 7")




#print("Step 8")


accuracy_logistic = count_regression*100/len(y_val)
        
print("Accurary using scikit learn: ",f'{accuracy_logistic:.2f}',"%")

#print("Step 9")

for i in range (len(nodes)):
    accuracy_mlp = count_mlp[i]*100/len(y_val)
    print("Accurary using MLP with ", nodes[i], " nodes each in 2 hidden layers: ",f'{accuracy_mlp:.2f}',"%")

#print("Step 10")