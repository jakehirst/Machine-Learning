from ast import List
from cProfile import label
from cmath import log
from math import log2
import numpy as np
import pandas as pd
from collections import Counter
    
    
'''
Reads the data from a csv. The csv has to have the attribute keys on the top row, and the labels in columns underneath their respecitve attributes.
'''
def Read_Data(filename):
    OrganizedData = {}
    df = pd.read_csv(filename)
    return df

'''
finds the entropy of a subset
possible outputs are the possible output values of the subset
'''
def Entropy(Subset, possible_outputs):
    i = 0
    p = []
    thingstosum = []
    whole = len(Subset)
    for output in possible_outputs:
        part = 0
        for row in Subset:
            if(row[len(row) -1] == output): part += 1
        p.append(part/whole)
        if(p[i] == 0 or p[i] == 1):
            return 0 #without this, log_2(0) or log_2(1) is NaN
        thingstosum.append(p[i]* log2(p[i]))
        i+=1

    entropy = -sum(thingstosum)
    return entropy

'''
gets the information gain of a single attribute based on the entropy

Attribute in this case is an integer from 0-3, referring to x_1 - x_3.
'''
def Information_gain(BigSubset, AttributeToTest_Index, AttributeToTest_possible_values, Possible_Outputs):
    thingstosum = []
    for value in AttributeToTest_possible_values:
        subset = []
        for row in BigSubset:
            if(row[AttributeToTest_Index] == value):
                subset.append(row)
        thingstosum.append((len(subset)/len(BigSubset)) * Entropy(subset, Possible_Outputs))
    return Entropy(BigSubset, Possible_Outputs) - sum(thingstosum)

'''
finds the attribute that has the best information gain out of the subset of data.

Subset = subset of the current node wishing to split
Attributes_Left = Attributes that the current node has not split on so far.
'''
def AttributeWithHighestInfoGain(Subset, Attributes_Left):
    if(len(Attributes_Left) == 1):
        return Attributes_Left[0]
    
    BestInfoGain = []
    Attribute_possible_values = set()
    Possible_Outputs = set()

    for Attribute in Attributes_Left:
        Attribute_possible_values.clear()
        Possible_Outputs.clear()
        
        attribute_index = Attributes_Left.index(Attribute)
        for row in range(1, len(Subset)):
            Attribute_possible_values.add(Subset[row][attribute_index])
            Possible_Outputs.add(Subset[row][len(Subset[row]) - 1])
            
        temp = Information_gain(Subset, attribute_index, Attribute_possible_values, Possible_Outputs)
        if(len(BestInfoGain) == 0):
            BestInfoGain.append(Attribute)
            BestInfoGain.append(temp)
        if(BestInfoGain[1] < temp):
            BestInfoGain[0] = Attribute
            BestInfoGain[1] = temp
    return BestInfoGain[0]  #returns the attribute with the best information gain, not the information gain.
     
     
'''
checks to see if all of the labels (outputs) of the subset are the same.

label_array is a numpy array from the subset df of the column of labels in the subset
'''
def All_Labels_Are_Da_Same(labelvals):
    unique_labels = np.unique(labelvals)
    if(len(unique_labels) == 0):
        return True, unique_labels[0]
    else:
        return False, None


'''
gets the most common label in the subset, and returns it.

Subset is the subset 
'''
def MostCommonLabel(labelvals):
    values, counts = np.unique(labelvals, return_counts=True)
    maxindex = counts.argmax()
    value = values[maxindex]
    count = counts[maxindex]
    return value, count

'''
returns a list of the attributes without the label key.
'''
def GetAttributesLeft(SubsetDict):
    ListOfAttributes = list(SubsetDict.keys())
    ListOfAttributes.pop()
    return ListOfAttributes

def GetValuesPossibleOfAttribute(DataFrame, AttributeToSplit):
    vals = np.array(DataFrame[AttributeToSplit])
    uniquevals = np.unique(vals)
    return uniquevals

'''
returns a new dataframe which is the same as the old dataframe, but removes the rows where
the attribute to split is not equal to val
'''
def SplitData(DataFrame, AttributeToSplit, val):
    newDF = DataFrame.loc[DataFrame[AttributeToSplit] == val ]
    del newDF[AttributeToSplit]
    return newDF
    

    
    



        
    
    

    
    
    