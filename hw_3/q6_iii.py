## CS 289 - HW3
## Lev Golod, 3032125890
## Question 3

import sys
import os
import numpy as np
import random
import numpy.linalg as lin
import scipy.io as sc
import math
from numpy import corrcoef, sum, log, arange
from numpy.random import rand

from pylab import pcolor, show, colorbar, xticks, yticks
import matplotlib.pyplot as plt

np.random.seed(20170120)

dir1="/Users/levgolod/GoogleDrive/Berkeley/02 Spring Semester/"
dir2="COMPSCI 289 - Machine Learning/homework/hw03/q6"
os.chdir(dir1+dir2)



##### get data - digits (MNIST)
##  NB first 784 cols are features. last col is labels
## train data is NxD, N = 60,000 D = 784
mnist = sc.loadmat('../data/hw3_mnist_dist/train.mat')['trainX']
N = mnist.shape[0]
D = mnist.shape[1] - 1
mnist_test = sc.loadmat('../data/hw3_mnist_dist/test.mat')['testX']

## contrast-normalize the pixel values.
## Divide each image (row) by its L2 norm
## Training Data
labs = mnist[:,-1,np.newaxis]
classes = np.unique(labs)
mnist = np.delete(mnist, -1, axis=1)

L2norms = np.apply_along_axis(lin.norm, 1, mnist)
L2norms.shape = (L2norms.shape[0], 1)
mnist = mnist / L2norms
## check that each row now has L2 norm of one
np.mean(np.isclose(np.apply_along_axis(lin.norm, 1, mnist),
 np.ones(mnist.shape[0])))
mnist = np.concatenate((mnist, labs), axis=1)
mnist[:,-1] = mnist[:,-1].astype(int)

## Testing Data
L2norms = np.apply_along_axis(lin.norm, 1, mnist_test)
L2norms.shape = (L2norms.shape[0], 1)
mnist_test = mnist_test / L2norms
## check that each row now has L2 norm of one
np.mean(np.isclose(np.apply_along_axis(lin.norm, 1, mnist_test),
 np.ones(mnist_test.shape[0])))


###########
#### Part C - Classification
####
###########

## Split data into training and validation sets
roworder = np.array(np.random.choice(mnist.shape[0],mnist.shape[0],False))
roworder.shape = (mnist.shape[0],1)
all_data = np.append(roworder, mnist,1)
mnist_val = all_data[all_data[:,0] < 10000]
mnist_val = np.delete(mnist_val, 0, axis=1)
mnist_train = all_data[all_data[:,0] >= 10000]
mnist_train = np.delete(mnist_train, 0, axis=1)
random.shuffle(mnist_train)
del(all_data)



##### (iv) - Prediction for Kaggle

## Lin Disc _xi fun - calculates Qc value for one class c, for one obs xi
## Based on formula provided in lecture notes
def lindisc_xi(xi, meanc, mycovinv, priorc, keepsc):
    term1 = np.dot( np.dot( meanc[keepsc], mycovinv ), xi[keepsc] )
    term2 = (-0.5) * np.dot( np.dot( meanc[keepsc], mycovinv ), meanc[keepsc] )
    term3 = math.log(priorc)
    return(sum((term1, term2, term3)))


X = mnist
# find estimated prior vector
priors = np.bincount(X[:,-1].astype(int))/X.shape[0]

## Loop over each class to get mean and cov matrix
means = [None] * len(classes)
covs = [None] * len(classes)
for c in classes:

    # print(' <> Class - ', c)
    # identify all observations in class c
    Xc = X[X[:,-1] == c]
    # delete last column - it contains class label
    Xc = np.delete(Xc, -1, axis=1)
    # Calculate mean vector - with D dimensions
    means[c] = np.mean(Xc, axis=0)
    # Calculate cov matrix - DxD
    covs[c] = np.cov(Xc, rowvar=False, bias=True)
    # Calculate pooled cov matrix - DxD
    if c == 0:
        # pooled_cov = np.cov(X, rowvar=False, bias = True)*X.shape[0]
        pooled_cov = covs[c]*Xc.shape[0]
    elif c > 0:
            pooled_cov = pooled_cov + covs[c]*Xc.shape[0]

## divide by N to finish the calculation -
## LDA: pooled within class cov matrix
pooled_cov = pooled_cov/(mnist_train.shape[0])

## Cov matrix is (presumably) singular so make adjustments
## this will be ok even if it's not
cov_prime = pooled_cov
print("cov mat (nearly) singular?" ,np.isclose( lin.det(cov_prime), 0 ))
# identify column numbers that DO/ DO NOT have all 0
# !!! consider playing with rtol to drop fewer dimensions /
#  maybe get smaller condition number
zeros = np.isclose(0, np.absolute(cov_prime).sum(axis=0), atol=5e-5)
print("proportion of dimensions dropped", sum(zeros)/len(pooled_cov))
if sum(zeros)/len(pooled_cov) > 0.85 :
    print("number of 0 cols in cov matrix worryingly high")
keeps = np.array(range(len(cov_prime)))[zeros != True]
drops = np.array(range(len(cov_prime)))[zeros]
# keep only those cols & rows from cov matrix that are not 0
cov_prime = np.delete(cov_prime, drops, axis=0)
cov_prime = np.delete(cov_prime, drops, axis=1)
# Find condition number of updated cov matrix
print("conditon num of updated cov matrix", \
lin.svd(cov_prime)[1][0]/lin.svd(cov_prime)[1][-1])
# calculate value for disc. fun: Sigma^-1
covinv = lin.inv(cov_prime)

## use apply-along to find vector of Qc values for each class c -
Qcs_test =[None] * len(classes)
Qcs_train =[None] * len(classes)
for c in classes:
    print("<><> Class : ", c)
    Qcs_train[c] = np.apply_along_axis(lindisc_xi, axis = 1, arr=mnist, \
    meanc=means[c], mycovinv=covinv, priorc=priors[c], keepsc=keeps)
    Qcs_test[c] = np.apply_along_axis(lindisc_xi, axis = 1, arr=mnist_test, \
    meanc=means[c], mycovinv=covinv, priorc=priors[c], keepsc=keeps)
Q_test = np.transpose(np.asarray(Qcs_test))
Q_train = np.transpose(np.asarray(Qcs_train))


## Make predictions
##  for each obs (row i) it is c that max lin disc fn (argmax col of Q for row i)
pred_test = np.argmax(Q_test, axis=1)
pred_train = np.argmax(Q_train, axis=1)
# pred_test = np.argmax(Q_test, axis=1)

## error rates
# sum(pred_train != X[:,-1])/len(X)

# outsheet csv
pred_test.shape = (pred_test.shape[0],1)
pred_test = pred_test.astype(int)
id = np.array(range(pred_test.shape[0])).astype(int)
id.shape = (id.shape[0],1)
answer = np.hstack((id, pred_test))
np.savetxt(fname='mnistpred.csv', X=answer, delimiter = ',', fmt='%.i', header = "Id, Category")
