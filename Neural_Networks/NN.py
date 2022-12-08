import numpy as np
import pandas as pd

#defining weight matrix,
# rows are the parent nodes it connects to
# columns are the child nodes it connects to
# layer is the layer
# w[layer][parent_node][child_node]
def define_hidden_weights(depth, width, initial_w="zeros"):
    depth = depth-1 #not including the output layer
    width = width+1 #including weight for bias term
    if(initial_w == "zeros"):
        return np.zeros(((width-1)*width) * depth).reshape(depth, width-1, width)
    elif(initial_w == "gaussian"):
        return np.random.normal(size=(width-1)*width*depth).reshape(depth, width-1, width)
    else:
        print("invalid initial w method")

        
def define_output_weights(width, initial_w):
    if(initial_w == "zeros"):
        return np.zeros(width+1)
    elif(initial_w == "gaussian"):
        return np.random.normal(size=(width+1))
    else:
        print("invalid initial w method")
        
def define_input_weights(input_width, width, initial_w):
    if(initial_w == "zeros"):
        return np.zeros((width)*input_width).reshape(width, input_width)
    elif(initial_w == "gaussian"):
        return np.random.normal(size=(width)*input_width).reshape(width, input_width)
    else:
        print("invalid initial w method")

#depth is the number of layers
#width is number of nodes per layer, NOT INCLUDING BIAS TERM
class NN():
    def __init__(self, depth, width, input_width, initial_w):
        self.hidden_weights = define_hidden_weights(depth-1, width, initial_w)
        self.output_weights = define_output_weights(width, initial_w)
        self.input_weights = define_input_weights(input_width, width, initial_w)
        self.depth = depth
        self.width = width

# x = NN(depth=3, width=2, initial_w="zeros")
# print(x.output_weights)
# print(x.hidden_weights)
# print("\n")
# x = NN(depth=3, width=2, initial_w="gaussian")
# print(x.output_weights)
# print(x.hidden_weights)
# print("\n")
# x = NN(depth=5, width=3, initial_w="zeros")
# print(x.output_weights)
# print(x.hidden_weights)
# print("\n")

        
        
