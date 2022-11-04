import sys
x = sys.path[0].split("/")
l = len(x)
last = x[l-1]
newpath = sys.path[0].rstrip(last)
sys.path.append(newpath)
sys.path.append(newpath + "Perceptron/")
import numpy as np
import pandas as pd
from Perceptron.standard_perceptron import *


header = ["variance", "skewness", "curtosis", "entropy", "label"]
dict = {0: 'variance',
        1: 'skewness',
        2: 'curtosis',
        3: 'entropy',
        4: 'label'}

training_df = pd.read_csv(newpath + "/Perceptron/train.csv", header=None)
test_df = pd.read_csv(newpath + "/Perceptron/test.csv", header=None)
# call rename () method
training_df.rename(columns=dict, inplace=True)
test_df.rename(columns=dict, inplace=True)


w = run_standard_perceptron(training_df, 0.01, 10)
test_error = test_w_with_dataframe(test_df, w)
print("--------  RESULTS  --------")
print("test error = " + str(test_error))
print("w = " + str(w))

