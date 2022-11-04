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

""" runs standard perceptron, starting with a w of zeros """
def run_standard_perceptron(training_df, r, T):
    w = np.zeros(len(training_df.columns))
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
            guess = np.matmul(w,feature_matrix[i])
            if(not np.sign(guess) == labels[i]):
                w = w + r * labels[i] * feature_matrix[i]
        #print("error = " + str(test_w(feature_matrix, labels, w)))
    return w


"""# able to use test_w() with a dataframe """
def test_w_with_dataframe(test_df, w):
    #change all of the 0 labels to -1
    for j in range(len(test_df)):
        if(test_df["label"][j] == 0):
            test_df['label'][j] = -1
    test_df.insert(0, 'b', np.ones(len(test_df)))
    feature_matrix = np.array(test_df.drop("label", axis=1))
    labels = np.array(test_df["label"])
    return test_w(feature_matrix, labels, w)


"""returns the percentage error against the given feature matrix and labels"""
def test_w(feature_matrix, labels, w):
    num_wrong = 0
    for i in range(len(labels)):
        guess = np.matmul(w,feature_matrix[i])  # guess = w^T*x
        if(not np.sign(guess) == labels[i]):
            num_wrong += 1
    return 100.0 * (num_wrong / len(labels))



    
        


        
        
