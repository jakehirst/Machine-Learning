import sys
x = sys.path[0].split("/")
l = len(x)
last = x[l-1]
newpath = sys.path[0].rstrip(last)
sys.path.append(newpath)
sys.path.append(newpath + "Perceptron")
import numpy as np
import pandas as pd
from Perceptron.averaged_perceptron import *
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


a = run_averaged_perceptron(training_df, 0.1, 10)
test_error = test_w_with_dataframe(test_df, a)

print("--------  RESULTS  --------")
print("test error = " + str(test_error))
print("a = ")
print(a)
# with pd.option_context("max_colwidth", 1000):
#     print (pd.DataFrame(w_array, columns=["weight vectors", "counts"]).to_latex())
# # print(pd.DataFrame(w_array, columns=["weight vectors", "counts"]).to_latex(index=False))



# doc = pl.Document()

# doc.packages.append(pl.Package('booktabs'))
# doc.packages.append(pl.Package('longtable'))

# with doc.create(pl.Section('Table: Global Faults ')):
#     doc.append(pl.NoEscape(latexdf.to_latex(longtable=True,caption='Five first')))

# print(doc)
# #doc.generate_pdf(filepath=f'Global_Fault', clean_tex=False)