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


###########
#### Part A - fit a Gaussian distribution to each DIGIT class
#### using maximum likelihood estimation
##########

## get data - digits (MNIST)
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

## split data based on classes
# np.unique(mnist[:,-1])
# np.bincount(mnist[:,-1].astype(int))/mnist.shape[0]

## Calculate MLE mean vector - with D dimensions (784)
## These lists will store the mean vectors and cov matrices for each class
mle_mean = [None] * len(classes)
mle_cov = [None] * len(classes)
for j in classes:
    # print('## ## ## Class - ', j)
    # identify all observations in class j
    X = mnist[mnist[:,-1] == j]
    # delete last column - it contains class label
    X = np.delete(X, -1, axis=1)
    # Calculate mean vector - with D dimensions
    mle_mean[j] = np.mean(X, axis=0)
    # Calculate cov matrix - DxD
    mle_cov[j] = np.cov(X, rowvar=False)

## Fit a Gaussian model for each digit class:
## The best-fitting Gaussian model for digit class j (based on MLE) is the one
# with mean vector mle_mean[j] and covariance matrix mle_cov[j]



###########
#### Part B - Visualize the covariance matrix for a particular class (digit).
#### How do the diagonal terms compare with the off-diagonal terms? What do you
#### conclude from this?
##########

fig = pcolor(mle_cov[0])
colorbar()
plt.suptitle('Question 6.b - Cov Matrix for Class [0]', fontsize=14, fontweight='bold')
plt.savefig('./plot/q6_b.png', bbox_inches='tight')
