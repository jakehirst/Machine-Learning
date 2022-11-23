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


#TODO: adjust gamma as needed
gamma_0 = 1e-3  
#TODO: adjust a as needed 
a = 1e-5   
C = [100/873, 500/873, 700/873]
df = get_df(newpath + "SVM/train.csv")
test_df = get_df(newpath + "SVM/test.csv")
w = []

for i in range(len(C)):
    w.append(stoch_sub_grad_desc(df, 100, C[i], learning_rate_method=2, gamma_0=gamma_0))

for i in range(len(C)):
    print("C = " + str(C[i]))
    print("w = " + str(w[i]))
    print("Training Error: ")
    print(get_error(df, w[i]))
    print("Test Error: ")
    print(get_error(test_df, w[i]))
          