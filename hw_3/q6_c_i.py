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
dir2="COMPSCI 289 - Machine Learning/homework/hw03/q6/mnist"
os.chdir(dir1+dir2)



##### get data - digits (MNIST)
##  NB first 784 cols are features. last col is labels
## train data is NxD, N = 60,000 D = 784
mnist = sc.loadmat('../../data/hw3_mnist_dist/train.mat')['trainX']
N = mnist.shape[0]
D = mnist.shape[1] - 1
mnist_test = sc.loadmat('../../data/hw3_mnist_dist/test.mat')['testX']

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



##### (i) - Linear Discriminant Analysis

## Lin Disc _xi fun - calculates Qc value for one class c, for one obs xi
## Based on formula provided in lecture notes
def lindisc_xi(xi, meanc, mycovinv, priorc, keepsc):
    term1 = np.dot( np.dot( meanc[keepsc], mycovinv ), xi[keepsc] )
    term2 = (-0.5) * np.dot( np.dot( meanc[keepsc], mycovinv ), meanc[keepsc] )
    term3 = math.log(priorc)
    return(sum((term1, term2, term3)))

## Loop over each different size of training subset
mnist_sizes = np.array([100,200,500,1000,2000,5000,10000,30000,50000])
# mnist_sizes = np.array([100,200])
error_train = [None] * len(mnist_sizes)
error_val = [None] * len(mnist_sizes)
for i in range(len(mnist_sizes)):

    print(" <><><>  <><><> ")
    print(" <><><> Size of training subset", mnist_sizes[i])
    print(" <><><>  <><><> ")
    # identify training subset
    X = mnist_train[:mnist_sizes[i],]
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

    ## identify and drop columns with 0 variance
    # from scipy import stats
    # stats.describe(cov_prime[np.diag_indices(cov_prime.shape[0])])
    zeros = np.isclose(0, cov_prime[np.diag_indices(cov_prime.shape[0])])
    drops1 = np.array(range(cov_prime.shape[0]))[zeros]
    keeps = np.array(range(len(cov_prime)))
    keeps = np.delete(keeps, drops1)
    cov_prime = np.delete(cov_prime, drops1, axis=0)
    cov_prime = np.delete(cov_prime, drops1, axis=1)

    ## get correlation matrix by dividing rows and cols by SDs
    sds = cov_prime[np.diag_indices(cov_prime.shape[0])]
    sds = np.power(sds, 0.5)
    cormat = cov_prime/sds[:,None]
    cormat = cormat/sds[None,:]

    np.allclose(cormat, np.transpose(cormat))
    np.mean(np.isclose(1, cormat[np.diag_indices(cormat.shape[0])]))==1
    # lin.cond(cormat)
    # lin.cond(cov_prime)

    # while Cov mat is singular, drop cols one by one
    drops2 = np.array(())
    j=0
    while ( 'result' not in vars() ):
        j = j +1
        ## try to find inverse
        try:
            result=lin.inv(cov_prime)
            break
        except:
            ## informal 'linear dependency score'
            ## L1 norm of each column of the correlation matrix
            score = np.sum( np.absolute(cormat), axis=0 )
            ## which element has highest score (break ties arbitrarily)?
            k=np.argmax(score)
            drops2 = np.append(drops2, k)
            # delete that row/col from Cov and Corr matrices
            cormat = np.delete(cormat, k, axis=0)
            cormat = np.delete(cormat, k, axis=1)
            cov_prime = np.delete(cov_prime, k, axis=0)
            cov_prime = np.delete(cov_prime, k, axis=1)

    keeps = np.delete(keeps, drops2)
    print("proportion of dimensions dropped", 1 - len(cov_prime)/len(pooled_cov))
    ## get inverse!
    # covinv = lin.inv(cov_prime)
    covinv = result

    ## use apply-along to find vector of Qc values for each class c -
    ##  - separately for training subset and val set
    Qcs_val =[None] * len(classes)
    Qcs_train =[None] * len(classes)
    Qcs_test =[None] * len(classes)
    for c in classes:
        print(" ## class", c)
        Qcs_train[c] = np.apply_along_axis(lindisc_xi, axis = 1, arr=X[:,:D], \
        meanc=means[c], mycovinv=covinv, priorc=priors[c], keepsc=keeps)
        Qcs_val[c] = np.apply_along_axis(lindisc_xi, axis = 1, \
        arr=mnist_val[:,:D], meanc=means[c], mycovinv=covinv, priorc=priors[c], \
        keepsc=keeps)
        # # pred for test set using biggest training subset
        # if mnist_sizes[i] == max(mnist_sizes):
        #     Qcs_test[c] = np.apply_along_axis(lindisc_xi, axis = 1, arr=mnist_test, \
        #     meanc=means[c], mycovinv=covinv, priorc=priors[c], keepsc=keeps)
    Q_val = np.transpose(np.asarray(Qcs_val))
    Q_train = np.transpose(np.asarray(Qcs_train))
    # Q_test = np.transpose(np.asarray(Qcs_test))

    ## Make predictions
    ##  for each obs (row i) it is c that max lin disc fn (argmax col of Q for row i)
    pred_val = np.argmax(Q_val, axis=1)
    pred_train = np.argmax(Q_train, axis=1)
    # pred_test = np.argmax(Q_test, axis=1)

    ## error rates
    error_val[i] = sum(pred_val != mnist_val[:,-1])/len(mnist_val)
    error_train[i] = sum(pred_train != X[:,-1])/len(X)

# print("Q6 C i : Train and Val Errors for LDA")
# print(mnist_sizes)
# print(error_train)
# print(error_val)


### Outsheet table with train/val error rates
mnist_sizes = np.array(mnist_sizes)
mnist_sizes.shape = (mnist_sizes.shape[0],1)

error_train = np.array(error_train)
error_train.shape = (error_train.shape[0],1)

error_val = np.array(error_val)
error_val.shape = (error_val.shape[0],1)

answer = np.hstack((mnist_sizes, error_train, error_val))
np.savetxt(fname='q6_partc_i_errors_lda.csv', X=answer, delimiter = ',', fmt='%.2f', \
header = "Subset size, Training Error, Validation Error")


### Plot the Training and Testing errors - LDA
plt.close()
plt.figure(1)
plt.scatter(range(len(mnist_sizes)), error_train)
plt.plot(range(len(mnist_sizes)), error_train)
plt.scatter(range(len(mnist_sizes)), error_val)
plt.plot(range(len(mnist_sizes)), error_val)
plt.xlabel('Size of training subset')
plt.ylabel('Prediction error')
plt.title('Question 6 (C) (i) : Prediction Accuracy', y=1.05)
plt.axis([0, len(mnist_sizes), 0, 1.1])
plt.xticks( range(len(mnist_sizes)), mnist_sizes)
plt.legend(['Training Error', 'Validation Error'], loc='center right')
plt.tight_layout()
plt.savefig('./plot/Question6_C_i_PredError_LDA.png')
