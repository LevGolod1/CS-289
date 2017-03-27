## CS 289 - HW1 
## Lev Golod

# cd GoogleDrive/Berkeley/02\ Spring\ Semester/
# cd COMPSCI\ 289\ -\ Machine\ Learning/homework/hw01/
# python hw01_q1_2.py

#NB - no hyper parameter tuning for Cifar

import scipy.io as sc
from scipy import stats
import numpy as np
import gc
from sklearn.svm import SVC
from sklearn import svm
from scipy import optimize
import copy
import pandas as pd
import timeit
import matplotlib.pyplot as plt

np.random.seed(20170120)

#### Question 1 - partition the mnist 

# ### myshuffle - a function to randomly shuffle the rows of a 2d np array 
# # X = copy.deepcopy(spam_train[:8,:4])
# # np.set_printoptions(precision=4)
# def myshuffle(X):
#     roworder = np.array(np.random.choice(X.shape[0],X.shape[0],False))
#     roworder.shape = (X.shape[0],1)
#     Y = pd.DataFrame(np.append(roworder, X, 1))
#     Y = Y.rename(columns = {0 : "roword"})
#     Y.sort_values("roword", inplace = True)
#     Z = Y.as_matrix()
#     Z = Z[:,1:]
#     return(Z)

# myshuffle(copy.deepcopy(spam_train[:8,:4]))



### MNIST - NB first 784 cols are features. last col is labels
mnist = sc.loadmat('hw01_data/mnist/train.mat')
mnist = mnist['trainX']  
mnist_test = sc.loadmat('hw01_data/mnist/test.mat')['testX']  
# create a vector of random digits with length = # of rows
# this is a random row order, which lets us shuffle the rows
roworder = np.array(np.random.choice(mnist.shape[0],mnist.shape[0],False))
roworder.shape = (mnist.shape[0],1)
all_data = np.append(roworder, mnist,1)
# use the random digits to partition into training and validation data
# NB: the row order becomes the 1st column 
# validation data is 1st 10000 shuffled rows. training is everything else
mnist_val = all_data[all_data[:,0] < 10000]
mnist_train = all_data[all_data[:,0] >= 10000]
del(mnist, roworder, all_data)

### SPAM - 1st 32 cols are features. last col is label
# NB - should coerce to integers
spam = sc.loadmat('hw01_data/spam/spam_data.mat')
spam_train = spam['training_data']
spam_labels = spam['training_labels']
spam_test = spam['test_data']
spam_train = np.append(spam_train, spam_labels.T,1)
del(spam_labels)
roworder = np.array(np.random.choice(spam_train.shape[0],spam_train.shape[0],
False))
roworder.shape = (spam_train.shape[0],1)
all_data = np.append(roworder, spam_train,1)
# all_data = myshuffle(all_data)
spam_val = all_data[all_data[:,0] <  roworder.shape[0]//5]
spam_train = all_data[all_data[:,0] >=  roworder.shape[0]//5]
spam = all_data
del(roworder, all_data)

### CIFAR - for training, 1st 3072 cols are features. 3073rd col is category
# - testing does not have categories given
cifar_train = sc.loadmat('hw01_data/cifar/train.mat')['trainX']
cifar_test = sc.loadmat('hw01_data/cifar/test.mat')['testX']
roworder = np.array(np.random.choice(cifar_train.shape[0],cifar_train.shape[0],
False))
roworder.shape = (cifar_train.shape[0],1)
all_data = np.append(roworder, cifar_train,1)
cifar_val = all_data[all_data[:,0] < 5000]
cifar_train = all_data[all_data[:,0] >= 5000]
del(roworder, all_data)



#### Question 2 - SVM

# Write a function to find the prediction error for training subset  
# and validation set. 
def svm_prederror(val_lab, val_feat, nsize, train, myC = 1, fn = 1):
  
    # initialize arrays to store error rates
    error_train = []
    error_val = []
    
    # loop over different training subset sizes
    for j in range(len(nsize)):
        
        # limit training subset to specified number of rows
        X = train[train[:,0] < nsize[j]]
         
        # last col of X is the vector of labels 
        y = X[:,-1]
    
        # get rid of 1st column (row numbers) and last (labels)
        X = np.delete(X, [0, X.shape[1]-1], 1)
         
        # fit the model, make predictions
        lin_clf = svm.LinearSVC(C=myC)
        if fn == 2:
             lin_clf = svm.SVC(C=myC, kernel = 'linear')
        lin_clf.fit(X, y) 
        pred_val = lin_clf.predict(val_feat)
        pred_train = lin_clf.predict(X)
         
        # training error, validation error 
        error_val.append(sum(val_lab != pred_val)/len(val_lab))
        error_train.append(sum(y != pred_train)/len(y))
    
    error_array = np.column_stack((error_train,error_val))
    return(error_array)


### MNIST
np.random.seed(20170120)
mnist_sizes = np.array([100,200,500,1000,2000,5000,10000])
mnist_errors = svm_prederror(val_lab = mnist_val[:,-1], 
              val_feat = np.delete(mnist_val, [0, mnist_val.shape[1]-1], 1),
              nsize = mnist_sizes + 10000,
              train = mnist_train)
              
print(mnist_errors)
print("--")

# plot the error rates
plt.figure(1)

# training subset 
plt.subplot(221)
plt.scatter(mnist_sizes, mnist_errors[:,0])
plt.plot(mnist_sizes, mnist_errors[:,0])
plt.xlabel('Size of training subset')
plt.ylabel('Prediction error')
plt.title('MNIST: Accuracy in Training Set', y=1.05)
plt.axis([0, mnist_sizes[-1], 0, 0.10])

plt.subplot(222)
plt.scatter(mnist_sizes, mnist_errors[:,1])
plt.plot(mnist_sizes, mnist_errors[:,1])
plt.xlabel('Size of training subset')
plt.ylabel('Prediction error')
plt.title('Prediction Accuracy in Validation Set', y=1.05)
plt.axis([0, mnist_sizes[-1], 0, 0.40])

plt.tight_layout()
plt.savefig('./plots/question2_PredAccur_mnist.png')


### SPAM
np.random.seed(20170120)
spam_sizes = np.array([100,200,500,1000,2000,5000,spam_train.shape[0]])
spam_errors = svm_prederror(val_lab = spam_val[:,-1], 
              val_feat = np.delete(spam_val, [0, spam_val.shape[1]-1], 1),
              nsize = spam_sizes + spam_val.shape[0],
              train = spam_train)

print(spam_errors)
print("--")

# plot the error rates
plt.close()
plt.figure(1)
xticks = list(range(0, spam_train.shape[0], 1000))

# training subset 
plt.subplot(221)
plt.scatter(spam_sizes, spam_errors[:,0])
plt.plot(spam_sizes, spam_errors[:,0])
plt.xlabel('Size of training subset')
plt.ylabel('Prediction error')
plt.title('Spam: Accuracy in Training Set', y=1.05)
plt.axis([0, spam_sizes[-1], 0, 0.40])
# plt.xticks(xticks,xticks,rotation='vertical')
plt.xticks(xticks,xticks)

plt.subplot(222)
plt.scatter(spam_sizes, spam_errors[:,1])
plt.plot(spam_sizes, spam_errors[:,1])
plt.xlabel('Size of training subset')
plt.ylabel('Prediction error')
plt.title('Prediction Accuracy in Validation Set', y=1.05)
plt.axis([0, spam_sizes[-1], 0, 0.40])
plt.xticks(xticks,xticks)

plt.tight_layout()
plt.savefig('./plots/question2_PredAccur_Spam.png')



### CIFAR 
np.random.seed(20170120)
cifar_sizes = np.array([100,200,500,1000,2000,5000])
cifar_errors = svm_prederror(val_lab = cifar_val[:,-1],
              val_feat = np.delete(cifar_val, [0, cifar_val.shape[1]-1], 1),
              nsize = cifar_sizes + 5000,
              train = cifar_train, fn = 2)

print(cifar_errors)
print("--")

# plot the error rates
plt.close()
plt.figure(1)
xticks = list(range(0, cifar_sizes[-1], 1000))

# training subset 
plt.subplot(221)
plt.scatter(cifar_sizes, cifar_errors[:,0])
plt.plot(cifar_sizes, cifar_errors[:,0])
plt.xlabel('Size of training subset')
plt.ylabel('Prediction error')
plt.title('CIFAR: Accuracy in Training Set', y=1.05)
plt.axis([0, cifar_sizes[-1], .5, 1])
# plt.xticks(xticks,xticks,rotation='vertical')
plt.xticks(xticks,xticks)

plt.subplot(222)
plt.scatter(cifar_sizes, cifar_errors[:,1])
plt.plot(cifar_sizes, cifar_errors[:,1])
plt.xlabel('Size of training subset')
plt.ylabel('Prediction error')
plt.title('Prediction Accuracy in Validation Set', y=1.05)
plt.axis([0, cifar_sizes[-1], .5, 1])
plt.xticks(xticks,xticks)

plt.tight_layout()
plt.savefig('./plots/question2_PredAccur_cifar.png')





#### Question 3

#NB: report the C values and respective accuracies, 
#  after ruling out implausible values (i.e. only report from rounds 2,3)

# NB:  only the 2nd round is really necessary. after that, improvements in  
#  error rate did not exceed 0.0001

# a function to return the validation error for the training subset of size n
def myerrorfn(c, my_n):
    return( svm_prederror(val_lab = mnist_val[:,-1], 
              val_feat = np.delete(mnist_val, [0, mnist_val.shape[1]-1], 1),
              nsize = my_n, train = mnist_train, myC = c)[-1,-1] )

# myerrorfn(1, np.array([200]) + 10000)
# myerrorfn(0.0001, np.array([200]) + 10000)
# myerrorfn(1, np.array([2000]) + 10000)

# rule out unreasonable values of C using small sample size: 200 
# myCs = 10.0**np.array([range(-5,10)])
myCs1 = 10.0**np.array(range(-10,10))
errors1 = []
for i in range(len(myCs1)):
    errors1.append(myerrorfn(myCs1[i], np.array([10000]) + 10000))

# which C value gave the best (smallest) error here? 
# Whatever that C is, we know we just need to try Cs within 1 order of  
#  magnitude in either direction.
bestC1 = myCs1[errors1.index(min(errors1))]


# do a 2nd round of optimization to get a tighter bound for best C
current = bestC1*0.1
myCs2 = np.array(current)
while (current < bestC1*10):
  current = current + (bestC1*10 - bestC1*0.1)*(3.0/100.0)
  myCs2 = np.append(myCs2, current)

errors2 = []
for i in range(myCs2.shape[0]):
    errors2.append(myerrorfn(myCs2[i], np.array([10000]) + 10000))

plt.scatter(myCs2, errors2)
plt.axis([myCs2[0], myCs2[-1], min(errors2)*0.9, max(errors2)*1.1])
plt.savefig('./plots/question3_r2c_errors.png')
# plt.show()


# # do a 3rd and final round of optimization using a training set w 10,000 rows
# index = errors2.index(np.array(errors2).min())
# bestC2 = myCs2[index]
# 
# min = myCs2[index-1]
# max = myCs2[index+1]
# step = (max - min)/20 
# current = min 
# myCs3 = np.array(current)
# while (current < max):
#     current = current + step 
#     myCs3 = np.append(myCs3, current)

# errors3 = []
# for i in range(myCs3.shape[0]):
#     errors3.append(myerrorfn(myCs3[i], np.array([10000]) + 10000))
# 
# index = errors3.index(np.array(errors3).min())
# bestC3 = myCs2[index]

# NB - create a data frame w 2 columns
allerrors = errors1 + errors2 
allCs = myCs1.tolist() + myCs2.tolist() 
Csanderrors = pd.DataFrame(list(zip(allCs, allerrors)))
Csanderrors = Csanderrors.rename(columns = {0 : "C", 1 : 'val_error'})
# Csanderrors =  Csanderrors.sort('val_error')
Csanderrors = Csanderrors.sort('val_error').reset_index(drop=True)
mnist_bestC =  Csanderrors.ix[0,0]
print("mnist_bestC", mnist_bestC)
print("--")
Csanderrors = Csanderrors.sort('C').reset_index(drop=True)
Csanderrors.to_csv('./tuneHyper_mnist.csv', index = False, header = True)

# ## Now that we have the cross-validated tuned hyperparameter, make prediction 
# ## MNIST - make prediction for test data
# X = copy.deepcopy(mnist_train)
# X = X[X[:,0] < 10000 + 10000]
# y = X[:,-1]
# 
# # get rid of 1st column (row numbers) and last (labels)
# X = np.delete(X, [0, X.shape[1]-1], 1)
#          
# # fit the model, make predictions
# lin_clf = svm.LinearSVC(C=mnist_bestC)
# lin_clf.fit(X, y) 
# pred_val = lin_clf.predict(mnist_test)
# 
# pred_df = pd.DataFrame(list(zip(range(len(pred_val)), pred_val)))
# pred_df.to_csv('./mnist_test_pred.csv', index = False, header = ('Id', 'Category'))

  
  
#### question 4 - k-fold validation 

# I have already shuffled my training data randomly, 
# by assigning random row-order indices from 1 to n, where n is the number
# of rows in the training data (5172). I will use these indices to split 
# the data into k = 5 'folds.' 

fold = (spam[:,0]//(spam.shape[0]/5)).astype(int)
# stats.itemfreq(fold)

spam_kfold = copy.deepcopy(spam)
spam_kfold[:,0] = fold
#stats.describe(spam[spam_kfold[:,0] == 0][:,0])

np.random.seed(20170120)

# function that returns validation error rate for 1 iteration of the k-fold
def myerrorfn2(c, my_val, my_train):
    return( svm_prederror(val_lab = my_val[:,-1], 
              val_feat = np.delete(my_val, [0, my_val.shape[1]-1], 1),
              nsize = np.array([100000]), train = my_train, myC = c)[-1,-1] )

# myerrorfn2(1, spam_kfold[spam_kfold[:,0] == 1],
# spam_kfold[spam_kfold[:,0] != 1])

# function that runs all k iterations of the k-fold and returns the
# average error rate
def mykfold(c): 
    error = []
    for j in range(5):
        my_val1 = spam_kfold[spam_kfold[:,0] == j]
        my_train1 = spam_kfold[spam_kfold[:,0] != j]
        error.append(myerrorfn2(c, my_val1, my_train1))
    return(sum(error)/5)


myCs1 = np.append(10.0**np.array(range(-5,5,1)), np.array(range(20,100,10)))
myCs1.sort()
errors1 = []
for i in range(len(myCs1)):
    errors1.append(mykfold(myCs1[i]))

# which C value gave the best (smallest) error here? 
# Whatever that C is, we know we just need to try Cs within 1 order of  
#  magnitude in either direction.
index = errors1.index(np.array(errors1).min())
bestC1 = myCs1[index]

kfold_Csdf = pd.DataFrame(list(zip(myCs1, errors1)))
kfold_Csdf = kfold_Csdf.rename(columns = {0 : "C", 1 : 'val_error'})
# kfold_Csdf =  kfold_Csdf.sort('val_error')
kfold_Csdf = kfold_Csdf.sort('val_error').reset_index(drop=True)
spam_bestC =  kfold_Csdf.ix[0,0]
kfold_Csdf = kfold_Csdf.sort('C').reset_index(drop=True)
kfold_Csdf.to_csv('./tuneHyper_Spam.csv', index = False, header = True)
print("spam_bestC", spam_bestC)
print("--")



#### Problem 5 - making predictions 

## Now that we have the cross-validated tuned hyperparameter, make prediction 
## Spam - make prediction for test data
X = copy.deepcopy(spam_train)
# X = myshuffle(X)
y = X[:,-1]
X = np.delete(X, [0, X.shape[1]-1], 1)
         
# fit the model, make predictions
lin_clf = svm.LinearSVC(C=spam_bestC)
lin_clf.fit(X, y) 
pred_val = lin_clf.predict(spam_test)

pred_df = pd.DataFrame(list(zip(range(len(pred_val)), pred_val)))
pred_df.to_csv('./spam_test_pred.csv', index = False, header = ('Id', 'Category'))


## Now that we have the cross-validated tuned hyperparameter, make prediction 
## MNIST - make prediction for test data
X = copy.deepcopy(mnist_train)
X = X[X[:,0] < 10000 + 10000]
y = X[:,-1]

# get rid of 1st column (row numbers) and last (labels)
X = np.delete(X, [0, X.shape[1]-1], 1)
         
# fit the model, make predictions
lin_clf = svm.LinearSVC(C=mnist_bestC)
lin_clf.fit(X, y) 
pred_val = lin_clf.predict(mnist_test)

pred_df = pd.DataFrame(list(zip(range(len(pred_val)), pred_val)))
pred_df.to_csv('./mnist_test_pred.csv', index = False, header = ('Id', 'Category'))      


# make prediction for Cifar test data
X = copy.deepcopy(cifar_train)
X = X[X[:,0] < 5000 + 5000]
y = X[:,-1]

# get rid of 1st column (row numbers) and last (labels)
X = np.delete(X, [0, X.shape[1]-1], 1)
         
# fit the model, make predictions
lin_clf = svm.SVC(kernel = 'linear')
lin_clf.fit(X, y) 
pred_val = lin_clf.predict(cifar_test)

pred_df = pd.DataFrame(list(zip(range(len(pred_val)), pred_val)))
pred_df.to_csv('./cifar_test_pred.csv', index = False, header = ('Id', 'Category'))

