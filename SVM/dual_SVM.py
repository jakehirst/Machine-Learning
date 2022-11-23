import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import math as m
from scipy.optimize import minimize
from stochastic_sub import *
import scipy

""" dual form of the optimization """
def alpha_eqsn(alpha, args):
    # args = [xi, yi, kernel_type="linear", gamma=None]
    if(args[2] == "linear"):
        return 1/2 * np.sum(np.outer(args[1],args[1]) * np.outer(alpha, alpha) * kernel_trick(args[2], args[0], args[0])) - np.sum(alpha)
    elif(args[2] == "gaussian"):
        return 1/2 * np.sum(np.outer(args[1],args[1]) * np.outer(alpha, alpha) * kernel_trick(args[2], args[0], args[0], args[3])) - np.sum(alpha)
    
""" This calculates the kernel trick. for the optimization for the dual form of SVM. We have either linear or gaussian choices here """
def kernel_trick(kernel_type, x, z, gamma=None):
    if(kernel_type == "linear"):
        return np.matmul(x,z.T) # linear = k(x,z) = x^T * z
    elif(kernel_type == "gaussian"):
        #https://www.dabblingbadger.com/blog/2020/2/27/implementing-euclidean-distance-matrix-calculations-from-scratch-in-python
        return np.exp(-(scipy.spatial.distance_matrix(x, z)**2) / gamma) #gaussian = k(x,z) = exp(-||x - z||^2 / gamma)


# def constraint(alpha, yi):
#     return np.sum(alpha*yi) == 0

""" Main running function to get the optimized alpha of the optimization equation in alpha_eqsn """
def get_alpha(df, C, kernel_type="linear", gamma=None):
    yi = np.array(df["label"])
    temp = pd.DataFrame(df)
    temp.drop('label', inplace=True, axis=1)
    xi = np.array(temp)
    initial_alpha = np.zeros(len(xi))
    
    #constraint on sum(alpha, yi) = 0
    func = lambda alpha: np.matmul(alpha, yi) 
    constraints = [{"type":"eq", "fun":func}] 
    
    tup = (0, C)
    bounds = np.array((tup, ) * len(xi))
    
    alpha = minimize(fun=alpha_eqsn, x0=initial_alpha, args=[xi,yi,kernel_type, gamma], method='SLSQP', bounds=bounds, constraints=constraints)
    #print(alpha)
    return alpha["x"]
    
""" gets the error of the learned weight vector on a given dataframe """
def get_error(df, alpha, w, beta, kernel_type="linear", gamma=None):
    y = np.array(df["label"])
    temp = pd.DataFrame(df)
    temp.drop('label', inplace=True, axis=1)
    x = np.array(temp)
    k = kernel_trick(kernel_type, x, x, 0.1)
    x = x.reshape(len(x[0]), len(x))
    w = w.reshape(1,len(w))
    predictions = np.sign(np.matmul(w,x) + beta)
    errors = 0
    for i in range(len(predictions)):
        if(y[i] == predictions[0][i]):
            continue
        else:
            errors += 1
    #print("error = " + str(errors/len(y)))
    return errors/len(y)

""" turns alpha into a weight vector """
def get_w(df, alpha, kernel_type="linear", gamma=None):
    y = np.array(df["label"])
    temp = pd.DataFrame(df)
    temp.drop('label', inplace=True, axis=1)
    x = np.array(temp)
    k = kernel_trick(kernel_type, x, x, 0.1)
    return np.sum((alpha*y).reshape(len(alpha), 1)* x, axis=0)

""" uses the weight vector learned from optimizing alpha into the bias term """
def get_beta(df, w):
    y = np.array(df["label"])
    temp = pd.DataFrame(df)
    temp.drop('label', inplace=True, axis=1)
    x = np.array(temp)
    w = w.reshape(len(w), 1)
    return np.sum(y.reshape(len(y), 1) - np.matmul(x,w)) / len(df)
    
    
# df = get_df("SVM/train.csv")
# alpha = get_alpha(df, 100/873, "gaussian", 0.1)
# w = get_w(df, alpha, "gaussian", 0.1)
# beta = get_beta(df, w)
# error = get_error(df, alpha, w, beta, "gaussian", 0.1)
# get_alpha(df, 100/873, "linear")
# print("done")


