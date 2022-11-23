import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import math as m


""" runs stochastic sub gradient descent """
def stoch_sub_grad_desc(df, max_epochs, C):
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
        r = get_learning_rate(epoch)
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
                deltaJ = w_0 - (C * N * yi * xi)
                print("delta J = " + str(deltaJ))
            else:
                w[1:] = (1-r) * w[1:]
                deltaJ = w_0
                print("delta J = " + str(deltaJ))
    return w

""" gets the learning rate (in notes its gamma_t"""
def get_learning_rate(t):
    if(t == 0):
        rate = 0.01
    elif(t == 1):
        rate = 0.005
    elif(t == 2):
        rate = 0.0025
    print("rate = " + str(rate))
    return rate   

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


data = np.array([[0.5, -1, 0.3, 1],
                     [-1, -2, -2, -1],
                     [1.5, 0.2, -2.5, 1]]) 
# labels = np.array([[1],
#                    [-1],
#                    [1]]) 

df = pd.DataFrame(data)
dict = {0: 'x0',
        1: 'x1',
        2: 'x2',
        3: 'label'}
df.rename(columns=dict, inplace=True)


stoch_sub_grad_desc(df, 3, 100/873)