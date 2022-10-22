import numpy as np
import pandas as pd
import random
import numpy.linalg
import math as m
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt




#reads the file and returns a pandas dataframe
def ReadFileAsDataFrame(filename):
    df = pd.read_csv(filename)
    return df
    
#trims the columns in colToTrim from the pandas dataframe df
def TrimColumns(df, colToTrim):
    return df.drop(colToTrim, axis=1)
    
#splits the dataset into training and test datasets
def SplitTrainingAndTesting(df, NumTestingExamples):
    x = random.sample(range(103), NumTestingExamples)
    TestDf = pd.DataFrame()
    for i in x:
        TestDf = TestDf.append(df.iloc[i])
    for i in x:
        df = df.drop(i)

    return TestDf.reset_index(), df.reset_index()

#finds the norm of the difference of the inputted weight vectors
def CheckTolerance(w_t, w_t_minus1):
    return np.linalg.norm(w_t - w_t_minus1)
    
# #saves the df to the filename to the specified path
# def SaveDataFrame(df, filepath):
#     print("not implemented yet")

#gets the error for a given row
def ErrorForRow(row, w_t, y_ColName):
    yi = row[y_ColName]
    xi = np.array(row.drop(y_ColName))
    guess = np.dot(w_t, xi)
    #print("guess = " + str(guess))
    #print("value = " + str(yi))
    error = yi - guess
    return error

#gets the gradient of the J(wt)
def Get_grad_J_wt(w_t, TrainingDf, row=None):
    if(not row == None):
        TrainingDf = TrainingDf.iloc[row]
    Grad_J_wt = np.zeros(len(w_t))
    sum = 0.0
    for j in range(0, len(w_t)):
        x_ijs = TrainingDf.drop("SLUMP(cm)", axis = 1)
        guess = np.matmul(w_t.T, x_ijs.T)
        error = TrainingDf["SLUMP(cm)"] - guess
        new_wj = np.matmul(error.T, TrainingDf.iloc[:,j])
        Grad_J_wt[j] = -new_wj
    
    return Grad_J_wt


#gets the gradient of J(wt) but only for one row
def Get_stochastic_grad_J_wt(w_t, TrainingDf, row=None):
    if(not row == None):
        TrainingRow = TrainingDf.iloc[row]
    Grad_J_wt = np.zeros(len(w_t))
    sum = 0.0
    for j in range(0, len(w_t)):
        x_ijs = TrainingRow.drop("SLUMP(cm)")
        guess = np.matmul(w_t.T, x_ijs.T)
        error = TrainingRow["SLUMP(cm)"] - guess
        new_wj = error * TrainingRow.iloc[j]
        Grad_J_wt[j] = -new_wj
    
    return Grad_J_wt
    

#gets the cost of the current weight vector
def GetCost(w_t, df):
    sum = 0.0
    for i, row in df.iterrows(): #go through each row in the df
        error = ErrorForRow(row, w_t, "SLUMP(cm)")
        sum += (error **2)

    cost = 0.5 * sum
    return cost


def RunStochasticGradientDecent(TrainingDf, tolerance_level, r):
    TrainingDf.insert(0, 'new-col', np.ones(len(TrainingDf)))
    TrainingDf.drop('index', inplace=True, axis=1)
    w_t = np.zeros(len(TrainingDf.columns) - 1)
    #x = np.array(TrainingDf.drop("SLUMP(cm)", axis=1))
    #y = np.array(TrainingDf["SLUMP(cm)"])
    #w = np.array([1,2,3,4,5,6,7,8])
    #DJ(x, y, w)
    w_t_minus1 = np.zeros(len(TrainingDf.columns) - 1)
    tolerance_achieved = False
    iteration = 0
    iterations = []
    norms = []
    costs = []
    RowNum = 0
    while(tolerance_achieved == False):
        if(RowNum > len(TrainingDf) -1):
            RowNum = 0
        #compute gradient of J_wt 
        Grad_J_wt = Get_stochastic_grad_J_wt(w_t, TrainingDf, RowNum)
        #update w_t to be w_t_plus1
        w_t_minus1 = w_t
        w_t = w_t_minus1 - r*Grad_J_wt
        
        #check to see if tolerance is achieved
        norm_btwn_w_vectors = CheckTolerance(w_t, w_t_minus1)
        if(norm_btwn_w_vectors < tolerance_level):
            tolerance_achieved = True
            converged = True
        elif((iteration == 5000) or m.isinf(norm_btwn_w_vectors)): #only let it go for 5000 iterations, or until norm_btwn_w_vectors reaches inf
            converged = False
            break
        else: 
            norms.append(norm_btwn_w_vectors)
            #print("iteration = " + str(iteration))
            #print(norm_btwn_w_vectors)
        iteration +=1
        iterations.append(iteration)
        cost = GetCost(w_t, TrainingDf)
        if(iteration % 100 == 0):
            print("\niteration: " + str(iteration) + " for r = " + str(r) +" cost =  " + str(cost))
        print(cost)
        costs.append(cost)
        RowNum +=1
        
    print("\ndone with r = " + str(r))
    print("converged = " + str(converged))
    print("w_t = " + str(w_t))
    print("MIN COST = " + str(min(costs)))
    return [w_t, iterations, norms, converged, costs, min(costs)]




def RunBatchedGradientDecent(TrainingDf, tolerance_level, r):
    TrainingDf.insert(0, 'new-col', np.ones(len(TrainingDf)))
    TrainingDf.drop('index', inplace=True, axis=1)
    w_t = np.zeros(len(TrainingDf.columns) - 1)
    #x = np.array(TrainingDf.drop("SLUMP(cm)", axis=1))
    #y = np.array(TrainingDf["SLUMP(cm)"])
    #w = np.array([1,2,3,4,5,6,7,8])
    #DJ(x, y, w)
    w_t_minus1 = np.zeros(len(TrainingDf.columns) - 1)
    tolerance_achieved = False
    iteration = 0
    iterations = []
    norms = []
    costs = []
    
    while(tolerance_achieved == False):
        #compute gradient of J_wt 
        Grad_J_wt = Get_grad_J_wt(w_t, TrainingDf)
        #update w_t to be w_t_plus1
        w_t_minus1 = w_t
        w_t = w_t_minus1 - r*Grad_J_wt
        
        #check to see if tolerance is achieved
        norm_btwn_w_vectors = CheckTolerance(w_t, w_t_minus1)
        if(norm_btwn_w_vectors < tolerance_level):
            tolerance_achieved = True
            converged = True
        elif((iteration == 5000) or m.isinf(norm_btwn_w_vectors)): #only let it go for 5000 iterations, or until norm_btwn_w_vectors reaches inf
            converged = False
            break
        else: 
            norms.append(norm_btwn_w_vectors)
            #print("iteration = " + str(iteration))
            #print(norm_btwn_w_vectors)
        iteration +=1
        iterations.append(iteration)
        cost = GetCost(w_t, TrainingDf)
        if(iteration % 100 == 0):
            print("\niteration: " + str(iteration) + " for r = " + str(r) +" cost =  " + str(cost))
        #print(cost)
        costs.append(cost)
    print("\ndone with r = " + str(r))
    print("converged = " + str(converged))
    print("w_t = " + str(w_t))
    print("MIN COST = " + str(min(costs)))
    return [w_t, iterations, norms, converged, costs, min(costs)]

def problem4a():
    TrainingFilename = "/Users/jakehirst/Downloads/slump_test.csv"
    initialDF = ReadFileAsDataFrame(TrainingFilename)
    colToTrim = ["No", "Compressive Strength (28-day)(Mpa)", "FLOW(cm)"]
    trimmedDf = TrimColumns(initialDF, colToTrim)
    TestDf, TrainingDf = SplitTrainingAndTesting(trimmedDf, 50)

    tolerance_level = 10**-6
    #r = 1.52587890625e-8
    r = 2.3e-8
    converged = RunBatchedGradientDecent(TrainingDf, tolerance_level, r)
    
    print("FINAL WEIGHT VECTOR = " + str(converged[0]))
    iterations = converged[1]
    costs = converged[4]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(iterations, costs, s=10, c='b', marker="s", label='cost vs iterations')
    ax1.set_ylabel("cost")
    ax1.set_xlabel("iterations")
    plt.legend(loc='upper right')
    plt.show()
    
def problem4b():
    TrainingFilename = "/Users/jakehirst/Downloads/slump_test.csv"
    initialDF = ReadFileAsDataFrame(TrainingFilename)
    colToTrim = ["No", "Compressive Strength (28-day)(Mpa)", "FLOW(cm)"]
    trimmedDf = TrimColumns(initialDF, colToTrim)
    TestDf, TrainingDf = SplitTrainingAndTesting(trimmedDf, 50)

    tolerance_level = 10**-6
    #r = 1.52587890625e-8
    r = 2.3e-8
    converged = RunStochasticGradientDecent(TrainingDf, tolerance_level, r)
    
    print("FINAL WEIGHT VECTOR = " + str(converged[0]))
    iterations = converged[1]
    costs = converged[4]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(iterations, costs, s=10, c='b', marker="s", label='cost vs iterations')
    ax1.set_ylabel("cost")
    ax1.set_xlabel("iterations")
    plt.legend(loc='upper right')
    plt.show()








if __name__ == "__main__":
    TrainingFilename = "/Users/jakehirst/Downloads/slump_test.csv"
    initialDF = ReadFileAsDataFrame(TrainingFilename)
    colToTrim = ["No", "Compressive Strength (28-day)(Mpa)", "FLOW(cm)"]
    trimmedDf = TrimColumns(initialDF, colToTrim)
    TestDf, TrainingDf = SplitTrainingAndTesting(trimmedDf, 50)

    problem4b()
    tolerance_level = 10**-6
    #r = 1.52587890625e-8
    r = 2.3e-8
    converged = RunBatchedGradientDecent(TrainingDf, tolerance_level, r)
    
    print("FINAL WEIGHT VECTOR = " + str(converged[0]))
    
    
    
    #setting my different learning rates
    rs = [0.001]
    for i in range(20):
        rs.append(rs[i] * 0.5)
    
    p = Pool(9)
    vals = []
    for i in range(len(rs)):
        val = p.apply_async(RunBatchedGradientDecent, [TrainingDf, tolerance_level, rs[i]])
        vals.append(val)
    for val in vals:
        val.wait()
    p.close()
    p.join()
    actual_values = []
    for val in vals:
        actual_values.append(val._value)
        
    
        
    
    print("done")
    
    