from NN import *
from Forward_propagation import *
from Backward_propagation import *
import os
import pandas as pd

def SGD(network, epochs, df, gamma_0, d):
    for epoch in range(epochs):
        learning_rate = gamma_0 / (1 + (gamma_0/d) * epoch)
        print("epoch = " + str(epoch))
        if(epoch > 1):
            print(get_error(df, network))

        #shuffle traiing set
        df = df.sample(
            frac=1,
            random_state=1
        ).reset_index(drop=True)
        
        for row in df.iterrows():            
            x = np.array(row[1].drop("label"))
            x = np.insert(x, 0, 1)
            true_y = row[1]["label"]
            y , z_matrix = forward_propogate(network, x)
            delta_output_weights, delta_hidden_weights, delta_input_weights = back_propogate(y, network, z_matrix, x, true_y=true_y)
            network.update_NN_weights(delta_output_weights, delta_hidden_weights, delta_input_weights, learning_rate)
        
        error = get_error(df, network)
        print(error)
        network.errors.append(error)
    return network
            
    
    
def get_error(df, network):
    errors = 0
    for row in df.iterrows():            
        x = np.array(row[1].drop("label"))
        x = np.insert(x, 0, 1)
        true_y = row[1]["label"]
        prediction = np.sign(forward_propogate(network, x)[0])
        if(prediction != true_y):
            errors += 1
    return errors/len(df)
        
    

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