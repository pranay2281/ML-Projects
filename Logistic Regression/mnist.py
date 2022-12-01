#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:07:31 2021

@author: pranay
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.datasets
import math


dataset = sk.datasets.fetch_openml('MNIST_784', data_home = "/Users/pranay/MNIST_home/")


X = dataset.data
labels = dataset.target

N=len(labels)

y=np.zeros(N)
for i in range(N):
    if labels[i]!='0':
        y[i]=1
        
N_train = 5000
        
i = np.random.randint(0,N_train)
img = X[i].reshape(28,28)
#plt.imshow(img,cmap = 'gray')

X_train = X[0:N_train]
X_val = X[N_train:]
Y_train = y[0:N_train]
Y_val = y[N_train:]

#standardize the data

X_train = X_train/255.0
X_val = X_val/255.0

#augmenting the matrix
X_train = np.insert(X_train, 0 , 1 ,axis=1)
X_val = np.insert(X_val,0,1,axis =1)



def sigmoid(u):
    return 1/(1 + np.exp(-u))


def cross_entropy(p, q):
    return (-p*np.log(q) - (1 - p)*np.log(1 - q))


def L(X, y, beta,gamma):
    
    N = np.shape(X)[0]
    
    L = 0
    
    for i in range(N):
        
        xi_aug = X[i]
        yi = y[i]
        
        L += cross_entropy(yi, sigmoid(xi_aug.T @ beta))
        
    L = L/N +  (0.5*gamma*(np.linalg.norm(beta)**2))

    return L

def grad_L(X, y, beta,gamma):
    
    N = np.shape(X)[0]
    d = np.shape(X)[1] - 1
    
    grad = np.zeros(d + 1)
    
    for i in range(N):
        
        xi_aug = X[i]
        yi = y[i]
        
        grad += (sigmoid(xi_aug.T @ beta) - yi)*xi_aug
        
    grad = grad/N + gamma*beta
    
    return grad



def fast_hessian (X, y , beta, gamma):
     N = np.shape(X)[0]
     d = np.shape(X)[1] - 1
     
     s_vals = sigmoid(X@beta)
     s_vals = np.reshape(s_vals, (N,1))
     M = (s_vals - s_vals**2)*X
     
     H = (1/N) * (X.T @ M) + gamma*np.identity(d+1)
     
     return H


N_val = X_val.shape[0]

d = np.shape(X_train)[1] - 1

t = 0.001
max_iter = 100
gamma = 10**-3


beta_n = np.zeros(d+1)
L_vals_n = []

for i in range(max_iter):
    
    L_vals_n.append(L(X_train, Y_train, beta_n,gamma))
     
    grad = grad_L(X_train, Y_train, beta_n,gamma)
    hessian = fast_hessian(X_train, Y_train, beta_n,gamma)
    
    beta_n = beta_n - t*np.linalg.solve(hessian, grad)
    
    #print(i , "newtons method")
    

plt.plot(L_vals_n, color="orange", label = "Newton's")

#testing with newton's method 
N_test = np.shape(X_val)[0]

y_pred = sigmoid(X_val @ beta_n)
count = 0

for j in range(N_test):
    
    if round(y_pred[j]) == Y_val[j]:
        count += 1
        
accuracy = count/N_test * 100
print("Accuracy:", accuracy, " with newton's method")

L_vals_sgd = []
num_epochs = 50
betak=np.zeros(d+1)

t=10**-7

for ep in range(num_epochs):
    #print(ep)
    L_vals_sgd.append(L(X_train,Y_train,betak,gamma))
    shuffled_idxs = np.random.permutation(X_train.shape[0])
    
    for i in shuffled_idxs:
        
        xiHat = X_train[i]
        yi = Y_train[i]
        
        grad = (sigmoid(xiHat.T @ betak) - yi)*xiHat
        
        betak = betak - t* grad
         
plt.plot(L_vals_sgd, color="green", label = "Stochastic")

#testing with stochastic gradient method 
N_test = np.shape(X_val)[0]

y_pred = sigmoid(X_val @ betak)
count = 0

for j in range(N_test):
    
    if round(y_pred[j]) == Y_val[j]:
        count += 1
        
accuracy = count/N_test * 100
print("Accuracy:", accuracy, " with stochastic gradient method")

gamma = 10**-3
t = 10**-7
max_iter = 500

beta_g = np.zeros(d+1)
L_vals_g=[]

for i in range (max_iter):
    
    L_vals_g.append(L(X_train,Y_train,beta_g, gamma))
    grad = grad_L(X_train, Y_train, beta_g,gamma)
    beta_g = beta_g -t*grad
    
    #print(i , "gradient descent")
    
plt.plot(L_vals_g,color = "red", label = "Gradient descent")

#testing with gradient descent method 
N_test = np.shape(X_val)[0]

y_pred = sigmoid(X_val @ beta_g)
count = 0

for j in range(N_test):
    
    if round(y_pred[j]) == Y_val[j]:
        count += 1
        
accuracy = count/N_test * 100
print("Accuracy:", accuracy, " with gradient descent")


plt.xlabel("Iterations")
plt.ylabel("L_vals")
plt.title("L_vals vs iterations")
plt.legend(loc="upper right")





    
    