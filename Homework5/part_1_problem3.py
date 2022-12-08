import numpy as np




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