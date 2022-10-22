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

if __name__ == "__main__":
    number3_bagging()
    number3_random_forest()
# def problem3_adaboost():
#     TestFileName = "Ensemble_Learning/number3_TestDf.csv"
#     filename = "Ensemble_Learning/number3_TrainingDf.csv"
#     columns_to_binarize = ["LIMIT_BAL", "AGE","BILL_AMT1","BILL_AMT3","BILL_AMT4","BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2" ,"PAY_AMT3" ,"PAY_AMT4","PAY_AMT5" ,"PAY_AMT6" ]
#     filenames = [filename, TestFileName]
#     stump = ID3.ID3()
#     data, Testdf = stump.prepData_quickly(filenames, howToFill='c', columns_to_binarize=columns_to_binarize)
#     weight_array = np.ones(len(data))
#     data = data.drop("Unnamed: 0", axis=1)
#     data = data.drop("index", axis=1)
#     data = data.assign(weights=weight_array)
#     weight_array = np.ones(len(Testdf))
#     Testdf = Testdf.drop("Unnamed: 0", axis=1)
#     Testdf = Testdf.drop("index", axis=1)
#     Testdf = Testdf.assign(weights=weight_array)

#     #stump.runID3(data, "Entropy", 1, data)
#     stumps_and_votes = AdaBoost(data, "Entropy", Testdf, 5)
    
#     data, Testdf = stump.prepData_quickly(filenames, howToFill='c', columns_to_binarize=columns_to_binarize)
#     weight_array = np.ones(len(data))
#     data = data.drop("Unnamed: 0", axis=1)
#     data = data.drop("index", axis=1)
#     data = data.assign(weights=weight_array)
#     weight_array = np.ones(len(Testdf))
#     Testdf = Testdf.drop("Unnamed: 0", axis=1)
#     Testdf = Testdf.drop("index", axis=1)
#     Testdf = Testdf.assign(weights=weight_array)  
      
#     p = Pool(8)
#     training_results = []
#     test_results = []
#     for num_stumps in range(1, len(stumps_and_votes)):
#         Training_numStumps_and_Error = p.apply_async(guess_rows, [data, stumps_and_votes, num_stumps])
#         training_results.append(Training_numStumps_and_Error)
#         Test_numStumps_and_Error = p.apply_async(guess_rows, [Testdf, stumps_and_votes, num_stumps])
#         test_results.append(Test_numStumps_and_Error)

#     for r in training_results:
#         r.wait()
#     p.close()
#     p.join()
    

#     training_T = []
#     training_error = []
#     print("TRAINING RESULTS")
#     for r in training_results:
#         print(r._value)
#         training_T.append(r._value[0])
#         training_error.append(r._value[1])

#     test_T = []
#     test_error = []
#     print("TEST RESULTS")
#     for test in test_results:
#         print(test._value)
#         test_T.append(test._value[0])
#         test_error.append(test._value[1])
        
        
#     #first plot
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)

#     ax1.scatter(test_T, test_error, s=10, c='b', marker="s", label='Test error')
#     ax1.scatter(training_T, training_error, s=10, c='r', marker="o", label='Training error')
#     plt.legend(loc='upper left')
#     plt.show()
    
    
#     #second plot
#     x = []
#     iteration = []
#     for i in range(len(stumps_and_votes)): 
#         x.append(stumps_and_votes[i][2])
#         iteration.append(i)
#     x = list(np.array(x)/100.0)
#     l = plt.figure()
#     ax2 = l.add_subplot(111)
#     ax2.scatter(iteration, x, s=10, c='b', marker="s", label='training errors per stump')
#     plt.legend(loc='upper right')
#     plt.show()
    
    
#     print("done")