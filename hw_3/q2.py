## CS 289 - HW3
## Lev Golod, 3032125890
## Question 2

import sys
import os

import matplotlib
import numpy as np
import math
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

dir1="/Users/levgolod/GoogleDrive/Berkeley/02 Spring Semester/"
dir2="COMPSCI 289 - Machine Learning/homework/hw03/q2"
os.chdir(dir1+dir2)

# References: http://matplotlib.org/examples/pylab_examples/contour_demo.html

## Write a function that takes the means and variances of two 2-dimensional
## Normal distributions, and outputs a contour plot

'Question 2, Part (c)'
'./plot/q2_c.png'

def drawcontour(title, file, mux1, muy1, varx1, vary1, cov1, \
 mux2 = 0, muy2 = 0, varx2 = 0, vary2 = 0, cov2 = 0):
    ## Identify the dimensions of the X,Y grid
    delta = 0.01
    xlb = (mux1 - mux2) - 4*math.pow(varx1+varx2, 0.5)
    xub = (mux1 - mux2) + 4*math.pow(varx1+varx2, 0.5)
    ylb = (muy1 - muy2) - 4*math.pow(vary1+vary2, 0.5)
    yub = (muy1 - muy2) + 4*math.pow(vary1+vary2, 0.5)
    x = np.arange(xlb, xub, delta)
    y = np.arange(ylb, yub, delta)
    X, Y = np.meshgrid(x, y)
    ## Calculate the densities of the Bivariate Normal distributions
    Z1 = mlab.bivariate_normal(X, Y, varx1, vary1, mux1, muy1, cov1)
    if varx2 != 0 and vary2 != 0:
        Z2 = mlab.bivariate_normal(X, Y, varx2, vary2, mux2, muy2, cov2)
        Z = Z1 - Z2
    else:
        Z = Z1
    ## Plot the contours
    plt.figure()
    CS = plt.contour(X, Y, Z, colors='k')
    # CS = plt.contour(X, Y, Z, colorscale='Jet')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(title)
    plt.savefig(file, bbox_inches='tight')

### Part A
drawcontour(title='Question 2, Part (a)', file='./plot/q2_a.png', \
mux1 = 1, muy1 = 1, varx1 = 1, vary1 = 2, cov1 = 0)

### Part B
drawcontour(title='Question 2, Part (b)', file='./plot/q2_b.png', \
mux1 = -1, muy1 = 2, varx1 = 2, vary1 = 3, cov1 = 1)

### Part C
drawcontour(title='Question 2, Part (c)', file='./plot/q2_c.png', \
 mux1 = 0, muy1 = 2, varx1 = 2, vary1 = 1, cov1 = 1, \
 mux2 = 2, muy2 = 0, varx2 = 2, vary2 = 1, cov2 = 1)

### Part D
drawcontour(title='Question 2, Part (d)', file='./plot/q2_d.png', \
 mux1 = 0, muy1 = 2, varx1 = 2, vary1 = 1, cov1 = 1, \
 mux2 = 2, muy2 = 0, varx2 = 2, vary2 = 3, cov2 = 1)

### Part E
drawcontour(title='Question 2, Part (e)', file='./plot/q2_e.png', \
 mux1 = 1, muy1 = 1, varx1 = 2, vary1 = 1, cov1 = 0, \
 mux2 = -1, muy2 = -1, varx2 = 2, vary2 = 2, cov2 = 1)
