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



TrainingFilename = "/Users/jakehirst/Desktop/Machine_Learning/DecisionTree/bank/train.csv"
TestFileName = "/Users/jakehirst/Desktop/Machine_Learning/DecisionTree/bank/test.csv"
columns_to_binarize = ["age", "balance","day","duration","campaign","pdays", "previous"]

    
m_prime = 1000 #number of examples in subsets for bagged trees
max_number_of_trees = 500
InfoGainMethod = "GiniIndex"
arr = [2,4,6]
for num_random_attributes in arr:
    trees = Get_Forest(TrainingFilename, max_number_of_trees, m_prime, InfoGainMethod, columns_to_binarize, num_random_attributes)

    start_time = time.time()
    TrainingDf = prep_subset(pd.read_csv(TrainingFilename), columns_to_binarize=columns_to_binarize)
    TestDf = prep_subset(pd.read_csv(TestFileName), columns_to_binarize=columns_to_binarize)
    print("DONE PREPPING DATAFRAMES")
    print("--- %s seconds ---" % (time.time() - start_time))
    
    start_time = time.time()
    columnTitles = np.array(list(TestDf.columns))
    Test_guesses = GuessesForAllTrees(TestDf, trees, columnTitles, 8)
    print("DONE WITH GUESSING TESTDF")
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    Training_guesses = GuessesForAllTrees(TrainingDf, trees, columnTitles, 8)
    print("DONE WITH GUESSING TRAINING DF")
    print("--- %s seconds ---" % (time.time() - start_time))
    
    start_time = time.time()
    test_errors = Compare_different_number_of_trees(Test_guesses, TestDf, cores=9)
    training_errors = Compare_different_number_of_trees(Training_guesses, TrainingDf, cores=9)
    print("DONE PRODUCING ERRORS")
    print("--- %s seconds ---" % (time.time() - start_time))

    xtest = []
    for i in range(len(test_errors)): 
        xtest.append(i+1)

        
    #first plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(xtest, test_errors, s=10, c='b', marker="s", label='Test error')
    plt.legend(loc='upper right')
    plt.show()
    
    
    #second plot
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(xtest, training_errors, s=10, c='r', marker="s", label='Training errors')
    plt.legend(loc='upper right')
    plt.show()
    
    print("done")
