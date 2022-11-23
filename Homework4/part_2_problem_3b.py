import sys
x = sys.path[0].split("/")
l = len(x)
last = x[l-1]
newpath = sys.path[0].rstrip(last)
sys.path.append(newpath)
sys.path.append(newpath + "SVM/")
import numpy as np
import pandas as pd
from SVM.stochastic_sub import *
from multiprocessing import Pool
from SVM.dual_SVM import *
from time import time

start = time()

C = [100/873, 500/873, 700/873]
df = get_df("SVM/train.csv")
test_df = get_df("SVM/test.csv")
gamma = [0.1, 0.5, 1, 5, 100]
alphas = []



print("\n\n--------- PART B AND C -------------")
for j in range(len(gamma)): 
    for i in range(len(C)):
        support_vectors = 0
        alpha = get_alpha(df, C[i], "gaussian", gamma[j])
        w = get_w(df, alpha, "gaussian", gamma[j])
        beta = get_beta(df, w)
        error = get_error(df, alpha, w, beta, "gaussian", gamma[j])
        test_error = get_error(test_df, alpha, w, beta, "gaussian", gamma[j])
        for k in range(len(alpha)):
            if(alpha[k] > 0):
                support_vectors += 1
        print("\ngamma = " + str(gamma[j]))
        print("C = " + str(C[i]))
        print("w = " + str(w))
        print("beta = " + str(beta))
        print("num support vectors = " + str(support_vectors))
        print("training error = " + str(error))
        print("test error = " + str(test_error))
        
        print("time = " + str(time() - start))


for j in range(len(gamma)):
    support_vectors = 0
    alpha = get_alpha(df, 500/873, "gaussian", gamma[j])
    alphas.append(alpha)
    w = get_w(df, alpha, "gaussian", gamma[j])
    beta = get_beta(df, w)
    error = get_error(df, alpha, w, beta, "gaussian", gamma[j])
    test_error = get_error(test_df, alpha, w, beta, "gaussian", gamma[j])
    for k in range(len(alpha)):
        if(alpha[k] > 0):
            support_vectors += 1
    print("\ngamma = " + str(gamma[j]))
    print("C = " + str(500/873))
    print("w = " + str(w))
    print("beta = " + str(beta))
    print("num support vectors = " + str(support_vectors))
    print("training error = " + str(error))
    print("test error = " + str(test_error))
    print("time = " + str(time() - start))

same_alphas = {}
for j in range(len(alphas)-1):
    alpha1 = alphas[j]
    alpha2 = alphas[j+1]
    num = 0
    for i in range(len(alpha1)):
        if(alpha1[i] == alpha2[i]):
            num += 1
    same_alphas[str(gamma[j]) + " , " + str(gamma[j+1])] = num

print(same_alphas)