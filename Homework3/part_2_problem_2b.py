import sys
x = sys.path[0].split("/")
l = len(x)
last = x[l-1]
newpath = sys.path[0].rstrip(last)
sys.path.append(newpath)
sys.path.append(newpath + "Perceptron")
import numpy as np
import pandas as pd
from Perceptron.voted_perceptron import *
import math
import pylatex as pl


header = ["variance", "skewness", "curtosis", "entropy", "label"]
dict = {0: 'variance',
        1: 'skewness',
        2: 'curtosis',
        3: 'entropy',
        4: 'label'}

training_df = pd.read_csv(newpath + "Perceptron/train.csv", header=None)
test_df = pd.read_csv(newpath + "Perceptron/test.csv", header=None)
# call rename () method
training_df.rename(columns=dict, inplace=True)
test_df.rename(columns=dict, inplace=True)


w_array = run_voted_perceptron(training_df, 0.1, 10)
test_error = test_w_with_dataframe(test_df, w_array)
for w in w_array:
    for i in range(len(w[0])):
        w[0][i] = round(w[0][i] * 1000) / 1000

latexdf = pd.DataFrame(w_array, columns=["weight vectors", "counts"])

print("--------  RESULTS  --------")
print("test error = " + str(test_error))
print("w = ")
# with pd.option_context("max_colwidth", 1000):
#     print (pd.DataFrame(w_array, columns=["weight vectors", "counts"]).to_latex())
# # print(pd.DataFrame(w_array, columns=["weight vectors", "counts"]).to_latex(index=False))
doc = pl.Document()

doc.packages.append(pl.Package('booktabs'))
doc.packages.append(pl.Package('longtable'))

with doc.create(pl.Section('Table: Global Faults ')):
    doc.append(pl.NoEscape(latexdf.to_latex(longtable=True,caption='Five first')))

print(doc)
#doc.generate_pdf(filepath=f'Global_Fault', clean_tex=False)