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

C = [100/873, 500/873, 700/873]
df = get_df("SVM/train.csv")
test_df = get_df("SVM/test.csv")

print("\n\n\n\n--------- PART A -------------")
for i in range(len(C)):
    alpha = get_alpha(df, C[i])
    w = get_w(df, alpha)
    beta = get_beta(df, w)
    error = get_error(df, alpha, w, beta)
    test_error = get_error(test_df, alpha, w, beta)
    print("\nC = " + str(C[i]))
    print("w = " + str(w))
    print("beta = " + str(beta))
    print("training error = " + str(error))
    print("test error = " + str(test_error))



