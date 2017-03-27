#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:28:45 2017

@author: levgolod
## CS 289 - HW4
## SID 3032125890
## Question 3
"""

#%% ##### Setup
import sys
import os
import numpy as np
import numpy.linalg as lin
import scipy.io as sc
from scipy import stats

np.random.seed(7530421)
dir1="/Users/levgolod/GoogleDrive/Berkeley/02 Spring Semester/"
dir2="COMPSCI 289 - Machine Learning/homework/hw04/q4/script"
os.chdir(dir1+dir2)




#%% ##### Get Data
datafile = sc.loadmat('../../data/data.mat')
X = datafile['X'] # main (training) data
X_test = datafile['X_test'] # testing data
y = datafile['y'].astype(int) # labels for training data
## Check dimensions 'n' stuff
X.shape
X_test.shape
y.shape
stats.describe(y)



#%% ##### Process Data - each feature should have mean 0, var 1
### Training data
X[:,0].shape
np.mean(X[:,0])
stats.describe(X[:,0])
## De-mean
#means = np.mean(X, axis=0, keepdims=True)
means = X.mean(axis=0)
means.shape
means[0]
X = X - means[np.newaxis, ]
stats.describe(X[:,0])
## Set var=1
vars = X.var(axis=0)
sds = vars**0.5
X = X/sds[np.newaxis, ]
stats.describe(X[:,3])
## Check
np.allclose(X.mean(axis=0), 0)
np.allclose(X.var(axis=0), 1)
## Column of ones
X.shape
X = np.append(X, np.ones((X.shape[0],1)), axis=1)
X.shape

### Testing data

## De-mean
means = X_test.mean(axis=0)
X_test = X_test - means[np.newaxis, ]
## Set var=1
vars = X_test.var(axis=0)
sds = vars**0.5
X_test = X_test/sds[np.newaxis, ]
## Check
np.allclose(X_test.mean(axis=0), 0)
np.allclose(X_test.var(axis=0), 1)
## Column of ones
X_test.shape
X_test = np.append(X_test, np.ones((X_test.shape[0],1)), axis=1)
X_test.shape




#%% ##### TRAIN/VAL SPLIT
## Design Matrix
roworder = np.array(np.random.choice(X.shape[0],X.shape[0],False))
roworder.shape = (X.shape[0],1)
all_data = np.append(roworder, X,1)
X_val = all_data[all_data[:,0] < 1000]
X_val = np.delete(X_val, 0, axis=1)
X_train = all_data[all_data[:,0] >= 1000]
X_train = np.delete(X_train, 0, axis=1)
del(all_data)
stats.describe(X[:,0])
stats.describe(X_train[:,0])
stats.describe(X_val[:,0])

## labels
all_y = np.append(roworder, y,1)
y_val = all_y[all_y[:,0] < 1000]
y_val = np.delete(y_val, 0, axis=1)
y_train = all_y[all_y[:,0] >= 1000]
y_train = np.delete(y_train, 0, axis=1)
del(all_y)
stats.describe(y[:,0])
stats.describe(y_train[:,0])
stats.describe(y_val[:,0])



### Choose a starting point using OLS
w0 = np.dot(
        lin.inv(np.dot(np.transpose(X_train), X_train)),
        np.dot(np.transpose(X_train), y_train)
        )
## how good is this? Pretty darn good
yhat = np.dot(X_train, w0)>0.5
yhat = yhat.astype(int)
accur_train = np.mean(yhat==y_train)
accur_train

yhat = np.dot(X_val, w0)>0.5
yhat = yhat.astype(int)
accur_val = np.mean(yhat==y_val)
accur_val
print(accur_train,accur_val)

## prediction for Kaggle
pred_test = np.dot(X_test, w0)>0.5
pred_test = pred_test.astype(int)

pred_test.shape = (pred_test.shape[0],1)
pred_test = pred_test.astype(int)
id = np.array(range(pred_test.shape[0])).astype(int)
id.shape = (id.shape[0],1)
answer = np.hstack((id, pred_test))
np.savetxt(fname='../out/wine_pred.csv', X=answer, delimiter = ',', fmt='%.i', header = 'Id,Category')
