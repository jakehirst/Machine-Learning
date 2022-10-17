from DecisionTree.ID3 import ID3
from DecisionTree.Function_Library import *
from Ensemble_Learning import *
import math as m
from multiprocessing import Pool
import matplotlib.pyplot as plt

def AdaBoost(data, InfoGainMethod, Testdf, T):
    total_weights = sum(data['weights'])
    data['weights'] = data['weights'].div(total_weights) #normalizing weights of the examples to sum to 1
    
    h = dict()
    for t in range(T):
        obj = ID3() #creating an ID3 object
        temp_h = obj.runID3(data, InfoGainMethod, 1, data)
        a_t = (m.log( (1 - temp_h[1]/100.0) / (temp_h[1]/100.0) )) / 2 #calculating vote (x[1])
        h[t] = [temp_h[0], a_t, temp_h[1]] #storing the rootNode for this stump and its vote in a dictionary and its training error
        print(f"a_t = {a_t}")
        print(f"temp_h = {temp_h}")
        data = update_weights(data, a_t, temp_h[0])
        obj = None
        temp_h = None
        print(f"t = {t}")
    return h
        
        

'''
makes a prediction of a dataset based off of the adaboost decision stumps and their votes
'''
def guess_rows(data, stumps_and_votes, num_stumps):
    error = 0.0
    for index, row in data.iterrows():
        guess = make_final_guess(stumps_and_votes, row, np.array(data.columns), num_stumps)
        if(guess == row['y']):
            continue
        else:
            error += row['weights']
    total_weights = sum(np.array(data['weights']))
    error = error / total_weights
    
    print(f"num_stumps = {num_stumps} and error = {error}")
    return [num_stumps, error]

'''
helper method for guess_rows. This guesses a single row based off of the adaboost decision stumps and their votes.
'''        
def make_final_guess(h, row, columnTitles, T):
    final_votes = dict()
    #print(f"T = {T}")
    for t in range(T):
        temp_guess = GuessLabel_4_Row(h[t][0], np.array(row), columnTitles)
        #print(f"temp guess = {temp_guess}")
        #print(f"vote = {h[t][1]}")
        if(not (final_votes.keys().__contains__(temp_guess))):
            final_votes[temp_guess] = h[t][1]
        else:
            final_votes[temp_guess] += h[t][1]
    
    #print(f"final votes = {final_votes}")  
    final_guess = max(final_votes, key=final_votes.get)
    return final_guess



'''
updates the weights of the dataset, in order to learn a new decision stump for adaboost.
'''
def update_weights(data, a_t, h_t):
    columnTitles = np.array(data.columns)
    #goes through each row to 
    for index, row in data.iterrows():
        if(row['y'] == GuessLabel_4_Row(h_t, np.array(row), columnTitles)): #if the answer from the stump is equal to the real answer...
            y_i_h_i = 1
        else:
            y_i_h_i = -1
        data.at[index,'weights'] = data.iloc[index]['weights'] * m.exp(-a_t * y_i_h_i)
    Z_t = sum(data['weights'])
    data['weights'] = data['weights'].div(Z_t)
    return data
    






if __name__ == '__main__':
    filename = "/Users/jakehirst/Desktop/Machine_Learning/DecisionTree/bank/train.csv"
    TestFileName = "/Users/jakehirst/Desktop/Machine_Learning/DecisionTree/bank/test.csv"
    columns_to_binarize = ["age", "balance","day","duration","campaign","pdays", "previous"]


    filenames = [filename, TestFileName]
    stump = ID3()
    data, Testdf = stump.prepData_quickly(filenames, 'unknown', 'c', columns_to_binarize)

    #stump.runID3(data, "Entropy", 1, data)
    stumps_and_votes = AdaBoost(data, "Entropy", Testdf, 500)
    
    data, Testdf = stump.prepData_quickly(filenames, 'unknown', 'c', columns_to_binarize)
    
    p = Pool(8)
    training_results = []
    test_results = []
    for num_stumps in range(1, len(stumps_and_votes)):
        Training_numStumps_and_Error = p.apply_async(guess_rows, [data, stumps_and_votes, num_stumps])
        training_results.append(Training_numStumps_and_Error)
        Test_numStumps_and_Error = p.apply_async(guess_rows, [Testdf, stumps_and_votes, num_stumps])
        test_results.append(Test_numStumps_and_Error)

    for r in training_results:
        r.wait()
    p.close()
    p.join()
    

    training_T = []
    training_error = []
    print("TRAINING RESULTS")
    for r in training_results:
        print(r._value)
        training_T.append(r._value[0])
        training_error.append(r._value[1])

    test_T = []
    test_error = []
    print("TEST RESULTS")
    for test in test_results:
        print(test._value)
        test_T.append(test._value[0])
        test_error.append(test._value[1])
        
        
    #first plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(test_T, test_error, s=10, c='b', marker="s", label='Test error')
    ax1.scatter(training_T, training_error, s=10, c='r', marker="o", label='Training error')
    plt.legend(loc='upper left')
    plt.show()
    
    
    #second plot
    x = []
    iteration = []
    for i in range(len(stumps_and_votes)): 
        x.append(stumps_and_votes[i][2])
        iteration.append(i)
    x = list(np.array(x)/100.0)
    l = plt.figure()
    ax2 = l.add_subplot(111)
    ax2.scatter(iteration, x, s=10, c='b', marker="s", label='training errors per stump')
    plt.legend(loc='upper right')
    plt.show()
    
    
    print("done")