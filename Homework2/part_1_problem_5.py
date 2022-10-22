import sys
x = sys.path[0].split("/")
l = len(x)
last = x[l-1]
newpath = sys.path[0].rstrip(last)
sys.path.append(newpath)
sys.path.append(newpath + "Linear_Regression")
from DecisionTree import ID3
from DecisionTree.Function_Library import *
from Ensemble_Learning import *
import math as m
from multiprocessing import Pool
import matplotlib.pyplot as plt
from Linear_Regression.LMS import *

def dJdw(x, y, w):
    dJdw = np.array([0,0,0,0])

    for j in range(len(w)):
        sum = 0
        for i in range(len(x)):
            error = y[i] - np.matmul(w.T,x[i])
            sum += error * x[i,j]
        dJdw[j] = sum
    return dJdw

def stochastich_update_w(x, y, w, r, i):
    new_w = np.zeros(len(w))
    for j in range(len(w)):
        error = y[i] - np.matmul(w.T,x[i])
        new_w[j] = w[j] + r*error*x[i,j]
    return new_w
    
    
    
if __name__ == "__main__":
    x = np.array([[1, 1,-1,2],
                  [1, 1,1,3],
                  [1, -1,1,0],
                  [1, 1,2,-4],
                  [1, 3,-1,-1]])
    y = np.array([1,4,-1,-2,0])
    b = -1
    w = np.array([b,-1,1,-1])
    
    dJdw = dJdw(x,y,w)
    print("\npart b: ")
    print("   dJdw = " +str(dJdw))
    print(f"   where dJdb = {dJdw[0]} (the first index of dJdw)")
    

    X = x.T
    XX_T_1 = np.linalg.inv(np.matmul(X,X.T))
    XY = np.matmul(X,y)
    optimal_w = np.matmul(XX_T_1, XY)
    print("\noptimal w for part c = ")
    print("   "+str(optimal_w) + "  where the optimal b is the first index of the array.\n")
    
    w = np.array([0,0,0,0])
    print("\npart d = ")
    for i in range(len(x)):
        print("   " + str(w))
        w = stochastich_update_w(x, y, w, 0.1, i)
    
    
    
    
    
    
    
    
    
    
    
