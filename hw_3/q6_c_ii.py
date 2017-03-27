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


##### (ii) - Quadratic Discriminant Analysis, QDA

## Quad Disc _xi fun - calculates Qc value for one class c, for one obs xi
## Based on formula provided in lecture notes
# xi = mnist_train[0, :D]
# meanc = means[1]
# covinvc = qda_covinvs[1]
# priorc = priors[c]
# detcovc = qda_covdets[1]
# quaddisc_xi(xi, meanc, covinvc, priorc, detcovc)
def quaddisc_xi(xi, meanc, covinvc, priorc, detcovc, keeps):
    xi_ctr = (xi-meanc)[keeps]
    term1 = (-0.5) * np.dot( np.dot(xi_ctr,covinvc), xi_ctr )
    try:
        term2 = (-0.5) * math.log(np.absolute(detcovc))
    except:
        term2 = 0
    term3 = math.log(priorc)
    return(sum((term1, term2, term3)))

## Loop over each different size of training subset
mnist_sizes = np.array([100,200,500,1000,2000,5000,10000,30000,50000])
# mnist_sizes = np.array([200])
error_train = [None] * len(mnist_sizes)
error_val = [None] * len(mnist_sizes)
ep = 1e-3 # value to 'perturb' singular values/vectors that are 0

##<><>indent1
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
    qda_covinvs = [None] * len(classes)
    qda_covdets = [None] * len(classes)
    qda_keeps = [None] * len(classes)
    qda_combine = [None] * len(classes)
    # Qc - the quadratic Discriminant function
    Qcs_val =[None] * len(classes)
    Qcs_train =[None] * len(classes)
    Qcs_test =[None] * len(classes)

    ##<><>indent2
    for c in classes:
        print(' <> Class - ', c)
        # identify all observations in class c
        Xc = X[X[:,-1] == c]
        # delete last column - it contains class label
        Xc = np.delete(Xc, -1, axis=1)
        Xprime = Xc
        # Calculate mean vector - with D dimensions
        means[c] = np.mean(Xc, axis=0)
        # Calculate cov matrix - DxD
        # QDA: different cov matrix for each class
        mycov = np.cov(Xc, rowvar=False, bias=True)
        # from scipy import stats
        # stats.describe(mycov.flatten())
        # stats.describe(Xc.flatten())

        ## Cov matrix is (presumably) singular so make adjustments
        ## this will be ok even if it's not
        cov_prime = mycov
        # diagonal elements of cov matrix 0? delete drop those dims
        try:
            diag0 = np.isclose(0, np.diag(cov_prime), atol=5e-5)
        except:
            qda_covinvs[c] = lin.pinv(cov_prime)
            print("bob")
            continue
        drops = np.array(range(len(cov_prime)))[diag0]
        qda_keeps[c] = np.array(list(set(np.array(range(len(mycov)))) - set(drops)))
        qda_combine[c] = drops
        cov_prime = np.delete(cov_prime, drops, axis=0)
        cov_prime = np.delete(cov_prime, drops, axis=1)
        keepdata = Xc[:,qda_keeps[c]]
        # combdata = sum(Xc[:,qda_combine[c]],axis=1)
        # combdata.shape = (combdata.shape[0], 1)
        # Xc_prime = np.hstack((keepdata, combdata))
        # cov_prime2 = np.cov(Xc_prime, rowvar=False, bias=True)

        U, s, V  = lin.svd(cov_prime)
        zeros = np.isclose(0, s)
        lindep = np.array(range(len(cov_prime)))[zeros]
        d = (lindep,lindep)
        add = np.zeros(cov_prime.shape)
        add[d] = np.random.uniform(0, 1, len(lindep))
        cov_prime = cov_prime + add

        k = 0
        while lin.det(cov_prime) < 1e-200:
            U, s, V  = lin.svd(cov_prime)
            zeros = np.isclose(0, s)
            lindep = np.array(range(len(cov_prime)))[zeros]
            d = (lindep,lindep)
            add = np.zeros(cov_prime.shape)
            add[d] = np.random.uniform(0, 1*k, len(lindep))
            cov_prime = cov_prime + add
            k = k + 1
            if k > 100:
                print(k, 'Problem')
                break

        lin.det(cov_prime)
        print("conditon num of updated cov matrix", lin.svd(cov_prime)[1][0]/lin.svd(cov_prime)[1][-1])
        # calculate value sfor discrim fn: Sigma_c^-1, det|Sigma_c|
        try:
            qda_covinvs[c] = lin.inv(cov_prime)
        except:
            qda_covinvs[c] = lin.pinv(cov_prime)
        # if determinant is too close to 0, assign an arbitrary small value
        if sum(np.isclose(lin.svd(cov_prime)[1],0)):
            qda_covdets[c] = 1e-15
        else:
            qda_covdets[c] = lin.det(cov_prime)

    ## use apply-along to find vector of Qc values for each class c -
    ##  - separately for training subset and val set
    for c in classes:
        Qcs_train[c] = np.apply_along_axis(quaddisc_xi, axis = 1, arr=X[:,:D], \
        meanc=means[c], covinvc = qda_covinvs[c], priorc=priors[c], detcovc=qda_covdets[c], keeps=qda_keeps[c])
        Qcs_val[c] = np.apply_along_axis(quaddisc_xi, axis = 1, arr=mnist_val[:,:D], \
        meanc=means[c], covinvc = qda_covinvs[c], priorc=priors[c], detcovc=qda_covdets[c], keeps=qda_keeps[c])
        ## pred for test set using biggest training subset
        if mnist_sizes[i] == max(mnist_sizes):
            Qcs_test[c] = np.apply_along_axis(quaddisc_xi, axis = 1, arr=mnist_test, \
            meanc=means[c], covinvc = qda_covinvs[c], priorc=priors[c], detcovc=qda_covdets[c], keeps=qda_keeps[c])

    ##<><>
    Q_val = np.transpose(np.asarray(Qcs_val))
    Q_train = np.transpose(np.asarray(Qcs_train))
    # Q_test = np.transpose(np.asarray(Qcs_test))

    ## Make predictions
    ##  for each obs (row i) it is c that max lin disc fn (argmax col of Q for row i)
    pred_val = np.argmax(Q_val, axis=1)
    pred_train = np.argmax(Q_train, axis=1)
    # pred_test_qda = np.argmax(Q_test, axis=1)

    ## error rates
    error_val[i] = sum(pred_val != mnist_val[:,-1])/len(mnist_val)
    error_train[i] = sum(pred_train != X[:,-1])/len(X)


# print("Q6 C ii : Train and Val Errors for QDA")
# print(mnist_sizes)
# print(error_train)
# print(error_val)


### Save out table with error rates
mnist_sizes = np.array(mnist_sizes)
mnist_sizes.shape = (mnist_sizes.shape[0],1)

error_train = np.array(error_train)
error_train.shape = (error_train.shape[0],1)

error_val = np.array(error_val)
error_val.shape = (error_val.shape[0],1)

answer = np.hstack((mnist_sizes, error_train, error_val))
np.savetxt(fname='q6_partc_ii_errors_qda.csv', X=answer, delimiter = ',', fmt='%.2f', \
header = "Subset size, Training Error, Validation Error")



### Plot the Training and Testing errors - QDA
plt.close()
plt.figure(1)
plt.scatter(range(len(mnist_sizes)), error_train)
plt.plot(range(len(mnist_sizes)), error_train)
plt.scatter(range(len(mnist_sizes)), error_val)
plt.plot(range(len(mnist_sizes)), error_val)
plt.xlabel('Size of training subset')
plt.ylabel('Prediction error')
plt.title('Question 6 (C) (ii) : Prediction Accuracy', y=1.05)
plt.axis([0, len(mnist_sizes), 0, 1.1])
plt.xticks( range(len(mnist_sizes)), mnist_sizes)
plt.legend(['Training Error', 'Validation Error'], loc='center right')
plt.tight_layout()
plt.savefig('./plot/Question6_C_ii_PredError_QDA.png')



##### (iii) - Which worked better? QDA or LDA
# I achieved about 20% validation error with QDA, and 13% with LDA. LDA worked
# better for me. I think this is because of the greater stability involved in
# calculating one common pooled-within-class covariance matrix for LDA, as
# opposed to 10 different class-specific covariance matrices with QDA. With QDA,
# each cov matrix had fewer observations to support it, and therefore had more
# linearly dependent elements. This meant I had to make larger adjustments to
# get the QDA matrices to be convertible.
