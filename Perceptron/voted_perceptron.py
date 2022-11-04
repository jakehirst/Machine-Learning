import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

''' changes the labels from  1 and 0  to  1 and -1 '''
def morph_labels(df):
    #change all of the 0 labels to -1
    for j in range(len(df)):
        if(df["label"][j] == 0):
            df['label'][j] = -1
    return df

""" runs voted perceptron, starting with a w of zeros """
def run_voted_perceptron(training_df, r, T):
    w_array = [[np.zeros(len(training_df.columns)), 0]] #array of all weight vectors w and their correct prediction counts
    m = 0 #number of weight vectors w
    Cm = 0 #number of correct predictions by a given weight vector w
    training_df.insert(0, 'b', np.ones(len(training_df)))
    
    #changing all the labels that are 0 to 1
    training_df = morph_labels(training_df)
    
    for t in range(T):
        #shuffle the data
        training_df = training_df.sample(frac=1).reset_index(drop=True)
        feature_matrix = np.array(training_df.drop("label", axis=1))
        labels = np.array(training_df["label"])
        for i in range(len(labels)):
            #only change w if the guess is not equal to the label
            guess = np.matmul(w_array[m][0],feature_matrix[i])
            if(not np.sign(guess) == labels[i]): #if the prediction is wrong...
                #print(np.sign(guess)*labels[i] <= 0) could replace the above if statement with this...
                w_array[m][1] = Cm #record how many predictions this weight vector got correct
                w_array.append([w_array[m][0] + r * labels[i] * feature_matrix[i], 0]) # make a new weight vector w and add it to the array
                m += 1 # start testing the new weight vector
                Cm = 1 #reset the number of examples examined by this weight vector
            else:
                Cm += 1 
    w_array[m][1] = Cm
    return w_array


"""# able to use test_w() with a dataframe """
def test_w_with_dataframe(test_df, w_array):
    #change all of the 0 labels to -1
    for j in range(len(test_df)):
        if(test_df["label"][j] == 0):
            test_df['label'][j] = -1
    test_df.insert(0, 'b', np.ones(len(test_df)))
    feature_matrix = np.array(test_df.drop("label", axis=1))
    labels = np.array(test_df["label"])
    return test_w(feature_matrix, labels, w_array)


"""returns the percentage error against the given feature matrix and labels"""
def test_w(feature_matrix, labels, w_array):
    num_wrong = 0
    for i in range(len(labels)):
        guess = 0.0
        for w in w_array: #loop to make guess on row based on all weight vectors
            guess += w[1] * np.sign(np.matmul(w[0],feature_matrix[i]))  
        if(not np.sign(guess) == labels[i]):
            num_wrong += 1
    return 100.0 * (num_wrong / len(labels))