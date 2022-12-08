import numpy as np
from NN import *
from Forward_propagation import sigmoid_func

def delta_sigmoid(s):
    return sigmoid_func(s) * (1 - sigmoid_func(s))

def back_propogate(y, network, z_matrix, input, true_y):
    output_weights = network.output_weights
    hidden_weights = network.hidden_weights
    input_weights = network.input_weights
    depth = network.depth
    width = network.width
    Loss = 0.5 * ((y - true_y)**2)

    dL_dy = y - true_y
    end_z = len(z_matrix) - 1
    #s =  output_weights[0] * z_matrix[end_z][0] + output_weights[1] * z_matrix[end_z][1] + ... + output_weights[n] * z_matrix[end_z][n]
    delta_output_weights = np.zeros(len(output_weights))
    for i in range(len(delta_output_weights)):
        delta_output_weights[i] = dL_dy * z_matrix[end_z][i]
    
    
    dL_dz_matrix = np.array([np.zeros(len(delta_output_weights)-1)])
    for i in range(len(dL_dz_matrix[0])):
        dL_dz_matrix[0][i] = dL_dy * output_weights[i+1]
    
    
    
    delta_hidden_weights = np.zeros(hidden_weights.shape)
    for layer in range(hidden_weights.shape[0], 0, -1):
        print(dL_dz_matrix)
        dL_dz_matrix = update_dL_dz_matrix(dL_dz_matrix, hidden_weights[layer-1], len(z_matrix[layer-1]))
        print(dL_dz_matrix)
        for parent in range(hidden_weights.shape[1]):
            for child in range(hidden_weights.shape[2]):
                if(layer == 0):
                    delta_hidden_weights[layer][parent][child] =  None
                else:
                    delta_hidden_weights[layer][parent][child] =  None
    

def update_dL_dz_matrix(dL_dz_matrix, hidden_weights, l, input_layer=False):
    new_row = np.array([np.zeros(l)])

    for i in range(1, len(dL_dz_matrix[0])):
        new_row[0][i] = np.matmul(dL_dz_matrix[0], hidden_weights[i])
        
    dL_dz_matrix = np.append(dL_dz_matrix, new_row, axis=0)
    return new_row
    # dL_dz_matrix.append()    
    
    