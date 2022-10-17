from DecisionTree.ID3 import ID3
from DecisionTree.Function_Library import *
from Ensemble_Learning import *
import math as m
from multiprocessing import Pool
import matplotlib.pyplot as plt

def partition_dataset(dataset, m_prime):
    
    
    



if __name__ == '__main__':
    filename = "/Users/jakehirst/Desktop/Machine_Learning/DecisionTree/bank/train.csv"
    TestFileName = "/Users/jakehirst/Desktop/Machine_Learning/DecisionTree/bank/test.csv"
    columns_to_binarize = ["age", "balance","day","duration","campaign","pdays", "previous"]

    filenames = [filename, TestFileName]
    stump = ID3()
    dataset, Testdf = stump.prepData_quickly(filenames, 'unknown', 'c', columns_to_binarize)
    m_prime = 
    
    partition_dataset(dataset, m_prime)
    