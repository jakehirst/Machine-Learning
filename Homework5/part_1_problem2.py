import numpy as np
import math as m

def sigmoid_func(node):
    return 1 / (1 + np.exp(-node))

def compute_node_vector(weights, node_vector, activation_function, output_layer=False):
    print("\nmatrix multiplication of " + str(weights.T) + " and " + str(node_vector))
    new_node_vector = np.matmul(weights.T, node_vector.T)
    print("result from matrix multiplication = " + str(new_node_vector))
    
    if(activation_function == "linear"):
        if(output_layer == True):
            print("output layer, dont add bias term:")
            return new_node_vector[0][0]
        new_node_vector = np.insert(new_node_vector, 0, 1, axis=0)#inserting bias parameter z_0
        return new_node_vector.T
    
    elif(activation_function == "sigmoid"):
        print("Using sigmoid function on this layer: ")
        new_node_vector = sigmoid_func(new_node_vector)
        new_node_vector = np.insert(new_node_vector, 0, 1, axis=0)#inserting bias parameter z_0
        print("new node vector = " + str(new_node_vector.T))
        return new_node_vector.T


#each row corresponds to an input node, and each column corresponds to the output node it is adding to
w1 = np.array([[-1, 1],
               [-2, 2],
               [-3, 3]])

w2 = np.array([[-1, 1],
               [-2, 2],
               [-3, 3]])

w3 = np.array([[-1],
               [2],
               [-1.5]])

input_x = np.array([[1,1,1]])


z1 = compute_node_vector(w1, input_x, "sigmoid")
z2 = compute_node_vector(w2, z1, "sigmoid")
y = compute_node_vector(w3, z2, "linear", output_layer=True)
print(y)






    