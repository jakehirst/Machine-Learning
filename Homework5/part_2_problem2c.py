import sys
x = sys.path[0].split("/")
l = len(x)
last = x[l-1]
newpath = sys.path[0].rstrip(last)
sys.path.append(newpath)
sys.path.append(newpath + "Neural_Networks/")
from Neural_Networks.SGD import *
import matplotlib.pyplot as plt

df = get_df(newpath + "SVM/train.csv")
test_df = get_df(newpath + "SVM/test.csv")

widths = [5, 10, 25, 50, 100]
for width in widths:
    network = NN(depth=3, width=width, input_width=5, initial_w="zeros")

    network = SGD(network, 10, df, 0.02, 0.07, test_df)

    number_of_updates = list(range(0, len(network.loss)))
    plt.plot(number_of_updates, network.loss, 'b') 
    plt.xlabel("number of updates")
    plt.ylabel("loss")
    plt.title(f"loss vs updates, width = {width}")
    plt.savefig(f"/Users/jakehirst/Desktop/Machine_Learning/Homework5/edited_width_{width}_zeros_loss.png")
    plt.close()
    number_of_epochs = list(range(0, len(network.errors)))
    plt.plot(number_of_epochs, network.errors, 'r')
    plt.xlabel("number of epochs")
    plt.ylabel("error")
    plt.title(f"error vs epochs, width = {width}")
    plt.savefig(f"/Users/jakehirst/Desktop/Machine_Learning/Homework5/edited_width_{width}_zeros_error.png")
    plt.close()
    
    plt.plot(number_of_epochs, network.test_errors, 'r')
    plt.xlabel("number of epochs")
    plt.ylabel("test_error")
    plt.title(f"testt_error vs epochs, width = {width}")
    plt.savefig(f"/Users/jakehirst/Desktop/Machine_Learning/Homework5/edited_width_{width}_zeros_test_error.png")
    plt.close()