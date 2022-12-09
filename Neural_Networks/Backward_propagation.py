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
    
    #going through all the hidden layer delta weights
    delta_hidden_weights = np.zeros(hidden_weights.shape)
    for layer in range(hidden_weights.shape[0], 0, -1):
        
        for parent in range(hidden_weights.shape[1]):
            for child in range(hidden_weights.shape[2]):
                
                s = np.matmul(hidden_weights[layer-1][parent], z_matrix[layer])
                dz_dw = delta_sigmoid(s)
                delta_hidden_weights[layer - 1][parent][child] = dL_dz_matrix[0][parent] * dz_dw * z_matrix[layer-1][child]
                print(delta_hidden_weights)
                print("\n")
        
        # print(dL_dz_matrix)
        # print(hidden_weights)
        dL_dz_matrix = update_dL_dz_matrix(dL_dz_matrix, hidden_weights[layer-1], z_matrix[layer])
        # print(dL_dz_matrix)
    
    # dL_dz_matrix = update_dL_dz_matrix(dL_dz_matrix, input_weights, input)
    #getting delta input weights
    delta_input_weights = np.zeros(input_weights.shape)
    for parent in range(len(z_matrix[0])-1):
        for input_num in range(input_weights.shape[1]):
            s = np.matmul(input_weights[parent], input)
            dz_dw = delta_sigmoid(s)
            delta_input_weights[parent][input_num] = dL_dz_matrix[0][parent] * dz_dw * input[input_num]
    
    print("\n")
    print(delta_output_weights)
    print(delta_hidden_weights)
    print(delta_input_weights)
    print("done")
    return delta_output_weights, delta_hidden_weights, delta_input_weights
        
                    
    

def update_dL_dz_matrix(dL_dz_matrix, hidden_weights, z_layer):
    num_parents = len(dL_dz_matrix[0])
    new_row = np.array([np.zeros(num_parents)])
    dz_dz = np.zeros(num_parents)
    
    for parent_count in range(num_parents):
        for parent_count_2 in range(num_parents):
            s = np.matmul(hidden_weights[parent_count_2], z_layer)
            delta_s = delta_sigmoid(s)
        dz_dz[parent_count] = delta_s * hidden_weights[parent_count][parent_count_2]
        new_row[0][parent_count] = np.matmul(dL_dz_matrix[0], dz_dz) # np.delete(hidden_weights[i], 0))
        
    #dL_dz_matrix = np.append(dL_dz_matrix, new_row, axis=0)
    print(new_row)
    return new_row
    # dL_dz_matrix.append()    
    
    