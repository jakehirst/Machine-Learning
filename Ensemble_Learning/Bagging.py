import sys
x = sys.path[0].split("/")
l = len(x)
last = x[l-1]
newpath = sys.path[0].rstrip(last)
sys.path.append(newpath)
import numpy as np
from DecisionTree.ID3 import ID3
from DecisionTree.Function_Library import *
from Ensemble_Learning import *
import math as m
from multiprocessing import Pool
import matplotlib.pyplot as plt
import random
import time


'''
Returns subset (dataframe) of dataset, with replacement and of size m_prime
'''
def get_random_subset(dataset, m_prime):
    random_list = [random.randint(0, len(dataset) - 1) for i in range(m_prime)]
    subset = pd.DataFrame()
    for index in random_list:
        subset = subset.append(dataset.iloc[index])
    subset = subset.reset_index(drop = True)
    return subset


def get_random_subsets(dataset, m_prime, cores, max_number_of_trees):
    p = Pool(cores)
    subsets = []
    
    for i in range(max_number_of_trees):
        sub = p.apply_async(get_random_subset, [dataset, m_prime])
        subsets.append(sub)
    
    for subset in subsets:
        subset.wait()
    p.close()
    p.join()
    p = None
    return subsets


def prep_subset(random_subset, MissingIndicator=None, howToFill=None, columns_to_binarize=None):
    #if there is a missing indicator and a howToFill method, then go ahead and fill the missing attributes.
    if((not MissingIndicator == None) and (not howToFill == None)):
        random_subset = FillMissingAttributes(random_subset, MissingIndicator, howToFill)
    #if there are columns to binarize, then go ahead and binarize those columns
    if(not columns_to_binarize == None):
        random_subset = binarize_numeric_vals(random_subset, columns_to_binarize)
    return random_subset

def prep_random_subsets(random_subsets, MissingIndicator=None, howToFill=None, columns_to_binarize=None, cores=1):
    prep_pool = Pool(cores) #change cores to decide how many cores to use in multiprocessing 
    
    prepped_data = []
    for random_subset in random_subsets:
        prepped_subset = prep_pool.apply_async(prep_subset, [random_subset._value, MissingIndicator, howToFill, columns_to_binarize])
        prepped_data.append(prepped_subset)
    for data in prepped_data:
        data.wait()
    prep_pool.close()
    prep_pool.join()
    return prepped_data


def findTree(subset, InfoGainMethod, MaxDepth, i):
    tree = ID3()
    tree = tree.ID3(subset, InfoGainMethod, MaxDepth=MaxDepth)
    if(i % 10 == 0):
        print(f"DONE WITH TREE # {i}")
    return tree


def findTrees(subsets, InfoGainMethod, max_number_of_trees, MaxDepth=100):
    p = Pool(9)
    trees = []
    for i in range(max_number_of_trees):
        data = subsets[i]._value
        tree = p.apply_async(findTree, [data, InfoGainMethod, MaxDepth, i])
        trees.append(tree)
    for tree in trees:
        tree.wait()
    p.close()
    p.join()
    p = None
    actual_trees = []
    for tree in trees:
        actual_trees.append(tree._value)
    return actual_trees

def GuessRow_with_n_trees(trees, row, columnTitles, n):
    guesses_for_row = []
    for i in range(n):
        temp_ans = GuessLabel_4_Row(trees[i], row, columnTitles)
        guesses_for_row.append(temp_ans)
    return guesses_for_row
    
def GuessesForAllTrees(TestDf, trees, columnTitles, cores):
    num_trees_testing = len(trees)
    guesses = []
    for i in range(len(TestDf)): #guessing each row in TestDf for all the trees
        guesses.append(GuessRow_with_n_trees(trees, TestDf.iloc[i], columnTitles, num_trees_testing))
    return guesses


def GuessEachRow(Test_guesses, numTrees, TestDf):
    #guesses = []
    error = 0
    for row in range(len(TestDf)):#go through each row
        row_guesses = Test_guesses[row][0:numTrees+1]
        amounts = np.unique(np.array(row_guesses), return_counts=True)
        IndexOfGuess = np.where(amounts[1] == max(amounts[1]))
        guess = amounts[0][IndexOfGuess]
        if(len(guess) > 1): #just in case there are equal votes for each label, just pick the first
            guess = guess[0]
        if(not (guess == TestDf.iloc[row]['y'])):
            error+=1
        #guesses.append(guess[0])
    return error/len(TestDf)

def Compare_different_number_of_trees(Test_guesses, TestDf, cores=8):
    p = Pool(cores)
    numTreesAndTheirErrors_temp = []
    for numTrees in range(len(Test_guesses[0])): #do this for each amount of trees
        #make a copy of the TestDf here?
        e = p.apply_async(GuessEachRow, [Test_guesses, numTrees, TestDf])
        numTreesAndTheirErrors_temp.append(e)
    for err in numTreesAndTheirErrors_temp:
        err.wait()
    p.close()
    p.join()
    p = None
    
    numTreesAndTheirErrors = []
    for err in numTreesAndTheirErrors_temp:
        numTreesAndTheirErrors.append(err._value)
    return numTreesAndTheirErrors
        
'''
This runs the algorithm that splits all of the training data into subsets and makes trees based off of those subsets.
'''
def GetBaggedTrees(TrainingFilename, max_number_of_trees, m_prime, InfoGainMethod, columns_to_binarize):
    start_time = time.time()
    OG_dataset = pd.read_csv(TrainingFilename)
    random_subsets = get_random_subsets(OG_dataset, m_prime, 8, max_number_of_trees)
    
    subsets = prep_random_subsets(random_subsets, MissingIndicator="unknown", howToFill="c", columns_to_binarize=columns_to_binarize, cores=9)
    print("DONE PREPPING SUBSETS")
    print("--- %s seconds ---" % (time.time() - start_time))
    
    start_time = time.time()
    trees = findTrees(subsets, InfoGainMethod, max_number_of_trees)
    print("DONE MAKING TREES")
    print("--- %s seconds ---" % (time.time() - start_time))
    return trees


def prep_number3_data():
    Filename = "/Users/jakehirst/Desktop/Machine_Learning/default_of_credit_card_clients.csv"
    df = pd.read_csv(Filename)
    NumTrainingExamples = 24000
    TrainingDf, df = SplitTrainingAndTesting(df, NumTrainingExamples)
    
    NumTestExamples = 6000
    TestDf, therest = SplitTrainingAndTesting(df, NumTestExamples)
    
    TrainingDf.to_csv('TrainingDf.csv')
    TestDf.to_csv('TestDf.csv')
    print("done")
    

#splits the dataset into training and test datasets
def SplitTrainingAndTesting(df, NumTestingExamples):
    x = random.sample(range(len(df)), NumTestingExamples)
    TestDf = pd.DataFrame()
    for i in x:
        TestDf = TestDf.append(df.iloc[i])
    for i in x:
        df = df.drop(i)

    return TestDf.reset_index(), df.reset_index()

def number3_bagging():
    TestFileName = "Ensemble_Learning/number3_TestDf.csv"
    TrainingFilename = "Ensemble_Learning/number3_TrainingDf.csv"

    columns_to_binarize = ["LIMIT_BAL", "AGE","BILL_AMT1","BILL_AMT3","BILL_AMT4","BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2" ,"PAY_AMT3" ,"PAY_AMT4","PAY_AMT5" ,"PAY_AMT6" ]

    m_prime = 1000 #number of examples in subsets for bagged trees
    max_number_of_trees = 500
    InfoGainMethod = "GiniIndex"
    
    trees = GetBaggedTrees(TrainingFilename, max_number_of_trees, m_prime, InfoGainMethod, columns_to_binarize)

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
    


def number3_random_forest():
    TestFileName = "Ensemble_Learning/number3_TestDf.csv"
    TrainingFilename = "Ensemble_Learning/number3_TrainingDf.csv"

    columns_to_binarize = ["LIMIT_BAL", "AGE","BILL_AMT1","BILL_AMT3","BILL_AMT4","BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2" ,"PAY_AMT3" ,"PAY_AMT4","PAY_AMT5" ,"PAY_AMT6" ]

    m_prime = 1000 #number of examples in subsets for bagged trees
    max_number_of_trees = 500
    InfoGainMethod = "GiniIndex"
    
    trees = Get_Forest(TrainingFilename, max_number_of_trees, m_prime, InfoGainMethod, columns_to_binarize)

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


        
# def run_bagging_algorithm(): #number 2 part b?
#     TrainingFilename = "/Users/jakehirst/Desktop/Machine_Learning/DecisionTree/bank/train.csv"
#     TestFileName = "/Users/jakehirst/Desktop/Machine_Learning/DecisionTree/bank/test.csv"
#     columns_to_binarize = ["age", "balance","day","duration","campaign","pdays", "previous"]

        
#     m_prime = 1000 #number of examples in subsets for bagged trees
#     max_number_of_trees = 5
#     InfoGainMethod = "GiniIndex"
    
#     trees = GetBaggedTrees(TrainingFilename, max_number_of_trees, m_prime, InfoGainMethod, columns_to_binarize)

#     start_time = time.time()
#     TrainingDf = prep_subset(pd.read_csv(TrainingFilename), columns_to_binarize=columns_to_binarize)
#     TestDf = prep_subset(pd.read_csv(TestFileName), columns_to_binarize=columns_to_binarize)
#     print("DONE PREPPING DATAFRAMES")
#     print("--- %s seconds ---" % (time.time() - start_time))
    
#     start_time = time.time()
#     columnTitles = np.array(list(TestDf.columns))
#     Test_guesses = GuessesForAllTrees(TestDf, trees, columnTitles, 8)
#     print("DONE WITH GUESSING TESTDF")
#     print("--- %s seconds ---" % (time.time() - start_time))

#     start_time = time.time()
#     Training_guesses = GuessesForAllTrees(TrainingDf, trees, columnTitles, 8)
#     print("DONE WITH GUESSING TRAINING DF")
#     print("--- %s seconds ---" % (time.time() - start_time))
    
#     start_time = time.time()
#     test_errors = Compare_different_number_of_trees(Test_guesses, TestDf, cores=9)
#     training_errors = Compare_different_number_of_trees(Training_guesses, TrainingDf, cores=9)
#     print("DONE PRODUCING ERRORS")
#     print("--- %s seconds ---" % (time.time() - start_time))

#     xtest = []
#     for i in range(len(test_errors)): 
#         xtest.append(i+1)

        
#     #first plot
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     ax1.scatter(xtest, test_errors, s=10, c='b', marker="s", label='Test error')
#     plt.legend(loc='upper right')
#     plt.show()
    
    
#     #second plot
#     fig2 = plt.figure()
#     ax2 = fig2.add_subplot(111)
#     ax2.scatter(xtest, training_errors, s=10, c='r', marker="s", label='Training errors')
#     plt.legend(loc='upper right')
#     plt.show()
    
#     print("done")
    













def find_RandomTree(subset, InfoGainMethod, num_random_attributes, MaxDepth, i):
    tree = ID3()
    tree = tree.randomID3(subset, InfoGainMethod, num_random_attributes, MaxDepth=MaxDepth)
    if(i % 10 == 0):
        print(f"DONE WITH TREE # {i}")
    return tree


def find_RandomTrees(subsets, InfoGainMethod, num_random_attributes ,max_number_of_trees, MaxDepth=100):
    p = Pool(9)
    trees = []
    for i in range(max_number_of_trees):
        data = subsets[i]._value
        tree = p.apply_async(find_RandomTree, [data, InfoGainMethod, num_random_attributes, MaxDepth, i])
        trees.append(tree)
    for tree in trees:
        tree.wait()
    p.close()
    p.join()
    p = None
    actual_trees = []
    for tree in trees:
        actual_trees.append(tree._value)
    return actual_trees


'''
This runs the algorithm that splits all of the training data into subsets and makes trees based off of those subsets.
'''
def Get_Forest(TrainingFilename, max_number_of_trees, m_prime, InfoGainMethod, columns_to_binarize, num_random_attributes):
    start_time = time.time()
    OG_dataset = pd.read_csv(TrainingFilename)
    random_subsets = get_random_subsets(OG_dataset, m_prime, 8, max_number_of_trees)
    
    subsets = prep_random_subsets(random_subsets, MissingIndicator="unknown", howToFill="c", columns_to_binarize=columns_to_binarize, cores=9)
    print("DONE PREPPING SUBSETS")
    print("--- %s seconds ---" % (time.time() - start_time))
    
    start_time = time.time()
    #trees = findTrees(subsets, InfoGainMethod, max_number_of_trees)
    trees = find_RandomTrees(subsets, InfoGainMethod, num_random_attributes, max_number_of_trees)

    print("DONE MAKING TREES")
    print("--- %s seconds ---" % (time.time() - start_time))
    return trees


def run_random_forest():
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

    



if __name__ == '__main__':
    run_random_forest()

    print("done")

    