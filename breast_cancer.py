#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 17:32:40 2021

@author: pranay
"""
import numpy as np
import sklearn as sk
import sklearn.datasets
import matplotlib.pyplot as plt
import sklearn.model_selection


dataset = sk.datasets.load_breast_cancer()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, train_size = 0.8)

#Augmenting X
X_train_aug = np.insert(X_train, 0, 1, axis = 1)
X_test_aug = np.insert(X_test, 0, 1, axis = 1)

N_train = np.shape(X_train_aug)[0]
N_test = np.shape(X_test_aug)[0]


def sigmoid(u):
    return (np.exp(u))/(1 + np.exp(u))


def cross_entropy(p, q):
    return (-p*np.log(q) - (1 - p)*np.log(1 - q))


def L(X, y, beta,gamma):
    
    N = np.shape(X)[0]
    
    L = 0
    
    for i in range(N):
        
        xi_aug = X[i]
        yi = y[i]
        
        L += cross_entropy(yi, sigmoid(xi_aug.T @ beta))
        #transpose dont matter
        
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


def hessian_L(X, y, beta, gamma):
    
    N = np.shape(X)[0]
    d = np.shape(X)[1] - 1
    
    hessian = np.zeros((d+1, d+1))
    
    for i in range(N):
        
        xi = X[i]
        ui = xi.T @ beta
        z = sigmoid(ui) - (sigmoid(ui))**2
        
        hessian += z * np.outer(xi, xi)
    
    hessian = hessian/N + gamma*np.identity(d+1)
    
    return hessian

d = np.shape(X_train_aug)[1] - 1

t = 0.001
max_iter = 500
gamma = 10**-3


beta_n = np.zeros(d+1)
L_vals_n = []

for i in range(max_iter):
    
    L_vals_n.append(L(X_train_aug, y_train, beta_n,gamma))
     
    grad = grad_L(X_train_aug, y_train, beta_n,gamma)
    hessian = hessian_L(X_train_aug, y_train, beta_n,gamma)
    
    beta_n = beta_n - t*np.linalg.solve(hessian, grad)
    
    #print(i , "newtons method")
    

plt.plot(L_vals_n, color="orange", label = "Newton's")


#testing with newton's method 
N_test = np.shape(X_test_aug)[0]

y_pred = sigmoid(X_test_aug @ beta_n)
count = 0

for j in range(N_test):
    
    if round(y_pred[j]) == y_test[j]:
        count += 1
        
accuracy = count/N_test * 100
print("Accuracy:", accuracy, " with newton's method")

L_vals_sgd = []
num_epochs = 30
betak=np.zeros(d+1)

t=10**-7

for ep in range(num_epochs):
    #print(ep)
    L_vals_sgd.append(L(X_train_aug,y_train,betak,gamma))
    shuffled_idxs = np.random.permutation(X_train_aug.shape[0])
    for i in shuffled_idxs:
        xiHat = X_train_aug[i]
        yi = y_train[i]
        
        grad = (sigmoid(xiHat.T @ betak) - yi)*xiHat
        
        betak = betak - t* grad
         
plt.plot(L_vals_sgd, color="green", label = "Stochastic")


#testing with stochastic method 
N_test = np.shape(X_test_aug)[0]

y_pred = sigmoid(X_test_aug @ betak)
count = 0

for j in range(N_test):
    
    if round(y_pred[j]) == y_test[j]:
        count += 1
        
accuracy = count/N_test * 100
print("Accuracy:", accuracy, " with stochastic gradient descent")

gamma = 10**-3
t = 10**-7
max_iter = 2500

beta_g = np.zeros(d+1)
L_vals_g=[]

for i in range (max_iter):
    
    L_vals_g.append(L(X_train_aug,y_train,beta_g, gamma))
    grad = grad_L(X_train_aug, y_train, beta_g,gamma)
    beta_g = beta_g -t*grad
    
    #print(i , "gradient descent")
    
plt.plot(L_vals_g,color = "red", label = "Gradient descent")

plt.xlabel("Iterations")
plt.ylabel("L_vals")
plt.title("L_vals vs iterations")
plt.legend(loc="upper right")

#testing with gradient descent method 
N_test = np.shape(X_test_aug)[0]

y_pred = sigmoid(X_test_aug @ beta_g)
count = 0

for j in range(N_test):
    
    if round(y_pred[j]) == y_test[j]:
        count += 1
        
accuracy = count/N_test * 100
print("Accuracy:", accuracy, " with gradient descent")
    


