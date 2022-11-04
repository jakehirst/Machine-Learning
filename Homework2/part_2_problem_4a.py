import sys
x = sys.path[0].split("/")
l = len(x)
last = x[l-1]
newpath = sys.path[0].rstrip(last)
sys.path.append(newpath)
sys.path.append(newpath + "Ensemble_Learning")
from DecisionTree import ID3
from DecisionTree.Function_Library import *
from Ensemble_Learning import *
import math as m
from multiprocessing import Pool
import matplotlib.pyplot as plt
from Ensemble_Learning.Adaboost import *
from Ensemble_Learning.Bagging import *
from Linear_Regression.LMS import *

if __name__ == "__main__":
    TrainingFilename = newpath + "Homework2/slump_test.csv"
    problem4a(TrainingFilename)