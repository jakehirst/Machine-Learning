import sys
x = sys.path[0].split("/")
l = len(x)
last = x[l-1]
newpath = sys.path[0].rstrip(last)
sys.path.append(newpath)
sys.path.append(newpath + "Neural_Networks/")
from Neural_Networks.Forward_propagation import *
from Neural_Networks.Backward_propagation import *

input = [1,1,1]
network = NN(depth=3, width=2, input_width=len(input), initial_w="gaussian")
network.hidden_weights = np.array([[[-1, -2, -3],
                                    [ 1,  2,  3]]])

network.input_weights = np.array([[-1, -2, -3],
                                   [ 1,  2,  3]])

network.output_weights = np.array([-1, 2, -1.5])
y, z_matrix = forward_propogate(network, input)
# print(y)
# print(z_matrix)
# print(network.output_weights)
# print(network.hidden_weights)
# print(network.input_weights)
print("done")
back_propogate(y, network, z_matrix, input, 1)