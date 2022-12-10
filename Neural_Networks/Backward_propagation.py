import numpy as np
from NN import *
from Forward_propagation import sigmoid_func

def delta_sigmoid(s):
    return sigmoid_func(s) * (1 - sigmoid_func(s))

#runs the back propogation algorithm to get the delta's of the new weight vectors for updating
def back_propogate(y, network, z_matrix, input, true_y):
    output_weights = network.output_weights
    hidden_weights = network.hidden_weights
    input_weights = network.input_weights
    depth = network.depth
    width = network.width
    Loss = 0.5 * ((y - true_y)**2)
    network.loss.append(Loss)

    dL_dy = y - true_y
    end_z = len(z_matrix) - 1

    delta_output_weights = np.zeros(len(output_weights))
    for i in range(len(delta_output_weights)):
        delta_output_weights[i] = dL_dy * z_matrix[end_z][i]
    
    dL_dz_matrix = np.array([np.zeros(len(delta_output_weights)-1)])
    for i in range(len(dL_dz_matrix[0])):
        dL_dz_matrix[0][i] = dL_dy * output_weights[i+1]
    
    #going through all the hidden layer delta weights
    delta_hidden_weights = np.zeros(hidden_weights.shape)
    for layer in range(hidden_weights.shape[0], 0, -1):
        for parent in range(hidden_weights.shape[1]):
            for child in range(hidden_weights.shape[2]):
                
                s = np.matmul(hidden_weights[layer-1][parent], z_matrix[layer])
                dz_dw = delta_sigmoid(s)
                delta_hidden_weights[layer - 1][parent][child] = dL_dz_matrix[0][parent] * dz_dw * z_matrix[layer-1][child]
        
        dL_dz_matrix = update_dL_dz_matrix(dL_dz_matrix, hidden_weights[layer-1], z_matrix[layer], hidden_weights.shape[0])

    #getting delta input weights
    delta_input_weights = np.zeros(input_weights.shape)
    for parent in range(len(z_matrix[0])-1):
        for input_num in range(input_weights.shape[1]):
            s = np.matmul(input_weights[parent], input)
            dz_dw = delta_sigmoid(s)
            delta_input_weights[parent][input_num] = dL_dz_matrix[0][parent] * dz_dw * input[input_num]

    return delta_output_weights, delta_hidden_weights, delta_input_weights
        
                    
    

def update_dL_dz_matrix(dL_dz_matrix, hidden_weights, z_layer, num_layers):
    num_parents = len(dL_dz_matrix[0])
    new_row = np.zeros(num_parents+1)
    dz_dz = np.zeros(num_parents)
    
    for i in range(hidden_weights.shape[0]):
        for j in range(hidden_weights.shape[1]):
            new_row[j] += dL_dz_matrix[0][i] * hidden_weights[i,j] \
                            * z_layer[i+1] * (1 - z_layer[i+1])
            #print(new_row)

    new_row = np.delete(new_row, 0)
    return np.array([new_row])
    
    