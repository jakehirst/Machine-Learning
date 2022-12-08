import numpy as np
from NN import *

def sigmoid_func(node):
    return 1 / (1 + np.exp(-node))


#z_matrix[layer][node]
def forward_propogate(network, input):
    depth = network.depth
    width = network.width
    hidden_weights = network.hidden_weights
    output_weights = network.output_weights
    input_weights = network.input_weights
    
    z_matrix = np.insert(np.zeros((depth-1) * width).reshape(depth-1, width), 0, np.ones(depth-1), axis=1)    
    for layer in range(depth-1):
        for node_num in range(1,width+1):
            if(layer == 0):
                z_matrix[layer][node_num] = sigmoid_func(np.matmul(input_weights[node_num-1], input))
            else:
                z_matrix[layer][node_num] = sigmoid_func(np.matmul(hidden_weights[layer-1][node_num-1], z_matrix[layer-1]))
                
    y = np.matmul(output_weights, z_matrix[depth-3])

    return y, z_matrix

