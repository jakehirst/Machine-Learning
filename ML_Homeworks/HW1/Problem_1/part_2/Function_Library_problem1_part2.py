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
def AttributeWithHighestInfoGain_Entropy(Subset, Attributes_Left):
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
finds the majority error of an attribute in the subset.
'''
def FindMajorityError(Subset, attribute_index):
    labelIdx = len(Subset[0]) - 1
    AttributeValues = np.unique(Subset[:, attribute_index])
    length_subset = Subset.shape[0]
    
    total_ME_Sv = 0.0
    for value in AttributeValues:
        Sv = []
        label_array = None
        labels_and_counts = None
        MajErr_Sv = None
        for row in Subset:
            if (row[attribute_index] == value):
                Sv.append(list(row))
        Sv = np.array(Sv)
        label_array = Sv[:, labelIdx]
        labels_and_counts = np.unique(label_array, return_counts=True)
        MajErr_Sv = (min(labels_and_counts[1])/sum(labels_and_counts[1])) 
        total_ME_Sv += ((len(Sv)/length_subset) * MajErr_Sv)
        
    return total_ME_Sv
    
'''
finds the attribute with the highest information gain using majority error.
'''
def AttributeWithHighestInfoGain_MajorityError(Subset, Attributes_Left):
    if(len(Attributes_Left) == 1):
        return Attributes_Left[0]
    
    labelIdx = len(Subset[0]) - 1
    label_array = Subset[:, labelIdx]
    labels_and_counts = np.unique(label_array, return_counts=True)
    MajErr_S = (min(labels_and_counts[1])/sum(labels_and_counts[1]))
    
    BestInfoGain = []
    Attribute_possible_values = set()
    Possible_Outputs = set()
    ME_Sv = 0.0
    for Attribute in Attributes_Left:
        attribute_index = Attributes_Left.index(Attribute)
        ME_Sv = FindMajorityError(Subset, attribute_index)
        
        temp_Gain = MajErr_S - ME_Sv
        if(len(BestInfoGain) == 0):
            BestInfoGain.append(Attribute)
            BestInfoGain.append(temp_Gain)
        if(BestInfoGain[1] < temp_Gain):
            BestInfoGain[0] = Attribute
            BestInfoGain[1] = temp_Gain
    return BestInfoGain[0]  #returns the attribute with the best information gain, not the information gain.
     
     
     
'''
finds the majority error of an attribute in the subset.
'''
def FindGiniIndex(Subset, attribute_index):
    labelIdx = len(Subset[0]) - 1
    AttributeValues = np.unique(Subset[:, attribute_index])
    length_subset = Subset.shape[0]
    
    total_GI_Sv = 0.0
    for value in AttributeValues:
        Sv = []
        label_array = None
        labels_and_counts = None
        GI_Sv = 1.0
        for row in Subset:
            if (row[attribute_index] == value):
                Sv.append(list(row))
        Sv = np.array(Sv)
        label_array = Sv[:, labelIdx]
        labels_and_counts = np.unique(label_array, return_counts=True)
        for count in labels_and_counts[1]:
            GI_Sv += -(count/sum(labels_and_counts[1]))**2
        total_GI_Sv += ((len(Sv)/length_subset) * GI_Sv)
        
    return total_GI_Sv
    
'''
finds the attribute with the highest information gain using majority error.
'''
def AttributeWithHighestInfoGain_GiniIndex(Subset, Attributes_Left):
    if(len(Attributes_Left) == 1):
        return Attributes_Left[0]
    
    labelIdx = len(Subset[0]) - 1
    label_array = Subset[:, labelIdx]
    labels_and_counts = np.unique(label_array, return_counts=True)
    GI_S = 1.0
    for i in range(len(labels_and_counts[1])):
        GI_S += -((labels_and_counts[1][i] / sum(labels_and_counts[1]))**2)
    
    BestInfoGain = []
    Attribute_possible_values = set()
    Possible_Outputs = set()
    GI_Sv = 0.0
    for Attribute in Attributes_Left:
        attribute_index = Attributes_Left.index(Attribute)
        GI_Sv = FindGiniIndex(Subset, attribute_index)
        
        temp_Gain = GI_S - GI_Sv
        if(len(BestInfoGain) == 0):
            BestInfoGain.append(Attribute)
            BestInfoGain.append(temp_Gain)
        if(BestInfoGain[1] < temp_Gain):
            BestInfoGain[0] = Attribute
            BestInfoGain[1] = temp_Gain
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

'''
checks the decision tree made against the tester data set.
'''
def CheckTreeAgainstTestData(TestFileName, rootNode, columnTitles):
    Testdf = pd.read_csv(TestFileName)
    print(Testdf)
    TestArray = Testdf.to_numpy()
    labelCol = len(columnTitles) -1

    correct = 0
    incorrect = 0
    rownum = 0
    for row in TestArray:
        rownum +=1
        labelFromtree = GuessLabel_4_Row(rootNode, row, columnTitles)
        labelFromRow = row[labelCol]
        if(labelFromtree == labelFromRow):
            correct += 1
        else:
            incorrect += 1
        
        print("Row number = " + str(rownum))
        print("Correct = " + str(correct))
        print("Incorrect = " + str(incorrect))
        print("Ratio = " + str(correct / (correct + incorrect)))
        

            
    
    
    print("\nresults are in")
    print("Row number = " + str(rownum))
    print("Correct = " + str(correct))
    print("Incorrect = " + str(incorrect))
    print("Percent Correct = " + str((correct / (correct + incorrect)) * 100.0))

'''
helper function for CheckTreeAgainstTestData()
checks to see if the row in the test data outputs the correct label 
'''
def GuessLabel_4_Row(rootNode, row, columnTitles):
    Isleaf = False
    node = rootNode
    label = None
    #until you find the leaf node...
    while(Isleaf == False):
        #get the attribute that the current node has split on
        attribute = list(node.info.keys())[0]
        print("*** "+ attribute + " ***")
        col = np.where(columnTitles == attribute)[0][0] #column index of the attribute
        value = row[col]
        print("value should be: " + str(value))
        #go to the next node until you find a leaf
        for i in range(len(node.info[attribute])):
            #going through each of the child nodes of the current node
            childNode = node.info[attribute][i]
            print(str(childNode.attributeVal))
            #if the newNode is a leaf and has the correct attributeVal, return the label 
            if(childNode.attributeVal == value and childNode.leaf == True):
                print("FOUND A LEAF")
                return childNode.label
                break
            #if the newNode 
            elif(childNode.attributeVal == value):
                print("found the right value")
                node = childNode
                break
            #if there is no node that has the attribute value that you are looking for, just return
            #the leaf node that has the most common label of the parent node.
            elif(i == len(node.info[attribute]) - 1):
                return node.info[attribute][0].label
        print(columnTitles)
        print(row)
    
    
    
    
    
    
    

    
    



        
    
    

    
    
    