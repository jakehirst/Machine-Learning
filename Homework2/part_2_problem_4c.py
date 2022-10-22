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
    TrainingFilename = "/Users/jakehirst/Downloads/slump_test.csv"
    initialDF = ReadFileAsDataFrame(TrainingFilename)
    colToTrim = ["No", "Compressive Strength (28-day)(Mpa)", "FLOW(cm)", "SLUMP(cm)"]
    x = np.array(TrimColumns(initialDF, colToTrim))
    y = np.array(initialDF["SLUMP(cm)"])

    X = x.T
    XX_T_1 = np.linalg.inv(np.matmul(X,X.T))
    XY = np.matmul(X,y)
    optimal_w = np.matmul(XX_T_1, XY)
    print("\noptimal w for part c = ")
    print("   "+str(optimal_w) + "  where the optimal b is the first index of the array.\n")
    
    
    
    