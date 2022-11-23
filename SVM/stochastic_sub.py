import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import math as m
# x = sys.path[0].split("/")
# l = len(x)
# last = x[l-1]
# newpath = sys.path[0].rstrip(last)

""" prepares the dataframe by adding the header and making binary 0 labels = -1 """
def get_df(filename):
    header = ["variance", "skewness", "curtosis", "entropy", "label"]
    dict = {0: 'variance',
            1: 'skewness',
            2: 'curtosis',
            3: 'entropy',
            4: 'label'}
    with open(os.path.join(filename), "r") as f:
        df = pd.read_csv(f, header=None)
    df.rename(columns=dict, inplace=True)
    for row in df.iterrows():
        if(row[1]["label"] == 0):
            df.at[row[0], 'label'] = -1
    return df
    
""" runs stochastic sub gradient descent """
def stoch_sub_grad_desc(df, max_epochs, C, learning_rate_method, gamma_0 ,a=None):
    w = np.zeros(len(df.columns))
    N = len(df)
    
    for epoch in range(max_epochs):
        print("\n***** EPOCH = " + str(epoch) + " *****")
        get_error(df, w) 
        
        #shuffling dataset
        df = df.sample(
            frac=1,
            random_state=1
        ).reset_index(drop=True)
        r = get_learning_rate(epoch, gamma_0, learning_rate_method, a)
        for row in df.iterrows():
            row = row[1]
            xi = np.array(row.drop("label"))
            #adding bias term to the end of xi
            xi = np.insert(xi, 0, 1)
            yi = row["label"]
            
            #making w_0 = w but with the bias parameter equal to 0.
            w_0 = np.array(w); w_0[0] = 0 
            
            if((np.matmul(w, xi) * yi) <= 1):
                w = w - r * w_0 +  r*C * N * yi * xi
            else:
                w[1:] = (1-r) * w[1:]
    return w

""" gets the error of the whole dataset"""
def get_error(df, w):
    errors = 0
    for row in df.iterrows():
        row = row[1]
        xi = np.array(row.drop("label"))
        #adding bias term to the end of xi
        xi = np.insert(xi, 0, 1.0)
        yi = row["label"]
        
        prediction = np.sign(np.matmul(xi, w))
        if(prediction == yi):
            continue
        else:
            errors += 1
    print("error = " + str(errors / len(df)))
                
                
""" gets the learning rate (in notes its gamma_t"""
def get_learning_rate(t, gamma_0, method, a=None):
    if(method == 1):
        rate = gamma_0 / (1 + (gamma_0 * t / a))
    elif(method == 2):
        rate = gamma_0 / (1 + t)
    print("rate = " + str(rate))
    return rate        
    
            


