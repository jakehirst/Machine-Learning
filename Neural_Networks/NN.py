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
        self.loss = []
        self.errors = []
        self.test_errors = []

        
    def update_NN_weights(self, delta_output_weights, delta_hidden_weights, delta_input_weights, learning_rate):
        self.hidden_weights = self.hidden_weights - learning_rate * delta_hidden_weights 
        self.output_weights = self.output_weights - learning_rate * delta_output_weights 
        self.input_weights = self.input_weights - learning_rate * delta_input_weights 



        
        
