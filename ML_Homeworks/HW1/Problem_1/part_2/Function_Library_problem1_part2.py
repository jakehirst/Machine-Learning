from ast import List
from cProfile import label
from cmath import log
from math import log2
import numpy as np
import pandas as pd
from collections import Counter
import pydot
import networkx as nx
import matplotlib.pyplot as plt

    
    
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
    whole = sum(np.transpose(np.array(Subset))[len(Subset[0])-1])
    for output in possible_outputs:
        part = 0
        for row in Subset:
            if(row[len(row) -2] == output): part += row[len(row) - 1]
        p.append(part/whole)
        if(p[i] == 0.0 or p[i] == 1.0):
            return 0.0 #without this, log_2(0) or log_2(1) is NaN
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
    #for each value in the Attribute's possible values, get all of the rows in the BigSubset that have that value 
    #in the respective attribute into a smaller subset, and sum the entropies of the small subsets
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
            Possible_Outputs.add(Subset[row][len(Subset[row]) - 2])
            
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
    labelIdx = len(Subset[0]) - 2
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
        labels = np.unique(label_array)
        
        #getting the majority error of Sv
        freq_of_labels = []
        for label in labels:
            num = 0.0
            for row in Sv:
                if(row[labelIdx] == label):
                    num += float(row[labelIdx+1])
            freq_of_labels.append(num)
        MajErr_Sv = (min(freq_of_labels)/sum(freq_of_labels))
        total_ME_Sv += ((len(Sv)/length_subset) * MajErr_Sv)
        
    return total_ME_Sv
    
'''
finds the attribute with the highest information gain using majority error.
'''
def AttributeWithHighestInfoGain_MajorityError(Subset, Attributes_Left):
    if(len(Attributes_Left) == 1):
        return Attributes_Left[0]
    
    labelIdx = len(Subset[0]) - 2
    label_array = Subset[:, labelIdx]
    The_labels = np.unique(label_array)

    #getting the majority error of the whole subset
    freq_of_labels = []
    for label in The_labels:
        num = 0.0
        for row in Subset:
            if(row[labelIdx] == label):
                num += row[labelIdx+1]
        freq_of_labels.append(num)
                
    MajErr_S = (min(freq_of_labels)/sum(freq_of_labels))
    
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
    labelIdx = len(Subset[0]) - 2
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
        labels = np.unique(label_array)
        whole = sum(np.transpose(np.array(Sv))[labelIdx+1].astype(float))
        GI_Sv = 1.0
        for label in labels:
            num = 0.0
            for row in Sv:
                if(row[labelIdx] == label):
                    num += float(row[labelIdx + 1])
            GI_Sv += -((num/whole)**2)
        total_GI_Sv += ((len(Sv)/length_subset) * GI_Sv)
        
    return total_GI_Sv
    
'''
finds the attribute with the highest information gain using majority error.
'''
def AttributeWithHighestInfoGain_GiniIndex(Subset, Attributes_Left):
    if(len(Attributes_Left) == 1):
        return Attributes_Left[0]
    
    labelIdx = len(Subset[0]) - 2
    label_array = Subset[:, labelIdx]
    labels = np.unique(label_array)
    whole = sum(np.transpose(np.array(Subset))[labelIdx+1])
    GI_S = 1.0
    for label in labels:
        num = 0.0
        for row in Subset:
            if(row[labelIdx] == label):
                num += float(row[labelIdx + 1])
        GI_S += -((num/whole)**2)
    
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
def MostCommonLabel(DataFrame, labelvals):
    possiblelabelvals = np.unique(np.array(labelvals))
    colNames = DataFrame.columns
    labelIndex = len(DataFrame.columns) - 2
    weightIndex = len(DataFrame.columns) - 1
    
    total_weights = []
    for val in possiblelabelvals:
        num = 0.0
        for row in DataFrame.iterrows():
            if(row[1][labelIndex] == val):
                num += row[1][weightIndex]
        total_weights.append(num)
    mcl = possiblelabelvals[total_weights.index(max(total_weights))]
    
    return mcl, max(total_weights)

'''
returns a list of the attributes without the label key.
'''
def GetAttributesLeft(SubsetDict):
    ListOfAttributes = list(SubsetDict.keys())
    ListOfAttributes.pop()
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
    labelCol = len(columnTitles) -2

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
  
    
    
'''
populates all missing attributes values based on the parameter howToFill
'''
def FillMissingAttributes(data, missingIndicator, howToFill):
    if(howToFill == 'a'):
        return FillWithMCA(data, missingIndicator)
    elif(howToFill == 'b'):
        return FillWithMCA_SameLabel(data, missingIndicator)
    elif(howToFill == 'c'):
        return FillWithFractionalCounts(data, missingIndicator)
    else:
        print("need to input different howTofill parameter")


'''
rows in the training data with missing feature values are completed by using the most common
value of the attribute among all examples
'''
def FillWithMCA(data, missingIndicator):
    weights = np.ones(len(data))
    #add ones to all the weights of the dataset
    data['weights'] = weights
    attributes = data.columns
    rows_with_missing = []
    #go through all of the rows of the dataset
    for index, row in data.iterrows():
        #go through each attribute in the row
        for attIDX in range(len(attributes)-2):
            #if there is a missing attribute value, drop it from the dataset and add it to the rows
            #with missing attribute values to be added in later with the correct attribute value
            if(row[attributes[attIDX]] == missingIndicator):
                rows_with_missing.append((row,attIDX))
                data = data.drop(index)
    data.reset_index()
    
    #getting all the new rows and their weights to be added back into the dataset
    newrows = []
    for row in rows_with_missing:
        columnarray = np.array(data._get_column_array(row[1]))
        vals_and_freq = np.unique(columnarray, return_counts=True)
        colval = attributes[row[1]]
        idx = np.where(vals_and_freq[1] == max(vals_and_freq[1]))[0][0]
        newrow = dict(row[0])
        newrow[colval] = vals_and_freq[0][idx]
        newrows.append(newrow)
    #add all of the new rows with the weights back to the dataset
    for row in newrows:
        data.loc[len(data.index)] = pd.Series(row)
    return data
        

'''
rows in the training data with missing feature values are completed by using the most common
value of the attribute among all examples with the same label
'''
def FillWithMCA_SameLabel(data, missingIndicator):
    weights = np.ones(len(data))
    #add ones to all the weights of the dataset
    data['weights'] = weights
    attributes = data.columns
    rows_with_missing = []
    #go through all of the rows of the dataset
    for index, row in data.iterrows():
        #go through each attribute in the row
        for attIDX in range(len(attributes)-2):
            #if there is a missing attribute value, drop it from the dataset and add it to the rows
            #with missing attribute values to be added in later with the correct attribute value
            if(row[attributes[attIDX]] == missingIndicator):
                rows_with_missing.append((row,attIDX))
                data = data.drop(index)
    data.reset_index()
    
    #getting all the new rows and their weights to be added back into the dataset
    newrows = []
    for row in rows_with_missing:
        data_same_label = data.loc[data[attributes[len(attributes)-2]] == row[0][attributes[len(attributes)-2]]]
        columnarray = np.array(data_same_label._get_column_array(row[1]))
        vals_and_freq = np.unique(columnarray, return_counts=True)
        colval = attributes[row[1]]
        idx = np.where(vals_and_freq[1] == max(vals_and_freq[1]))[0][0]
        newrow = dict(row[0])
        newrow[colval] = vals_and_freq[0][idx]
        newrows.append(newrow)

    #add all of the new rows with the weights back to the dataset
    for row in newrows:
        data.loc[len(data.index)] = pd.Series(row)
    return data
    
'''
rows in the training data with missing feature values are completed by adding fractoinal counts
of the attribute values in the training data
'''
def FillWithFractionalCounts(data, missingIndicator):
    weights = np.ones(len(data))
    #add ones to all the weights of the dataset
    data['weights'] = weights
    attributes = data.columns
    rows_with_missing = []
    #go through all of the rows of the dataset
    for index, row in data.iterrows():
        #go through each attribute in the row
        for attIDX in range(len(attributes)-2):
            #if there is a missing attribute value, drop it from the dataset and add it to the rows
            #with missing attribute values to be added in later with the correct attribute value
            if(row[attributes[attIDX]] == missingIndicator):
                rows_with_missing.append((row,attIDX))
                data = data.drop(index)
    data.reset_index()
    
    #getting all the new rows and their weights to be added back into the dataset
    newrows = []
    for row in rows_with_missing:
        columnarray = np.array(data._get_column_array(row[1]))
        vals_and_freq = np.unique(columnarray, return_counts=True)
        colval = attributes[row[1]]
        for i in range(len(vals_and_freq[0])):
            attval = vals_and_freq[0][i]
            newrow = dict(row[0])
            newrow[colval] = attval
            newrow['weights'] = vals_and_freq[1][i]/sum(vals_and_freq[1])
            newrows.append(newrow)

    #add all of the new rows with the weights back to the dataset
    for row in newrows:
        data.loc[len(data.index)] = pd.Series(row)
    return data
    












def visualize(rootNode,graph):
    root = pydot.Node("val = " + rootNode.attributeVal + '\n splitting on: ' + list(rootNode.info.keys())[0], 
                      label="val = " + rootNode.attributeVal + "\n splitting on: " + list(rootNode.info.keys())[0])
    graph.add_node(root)
    for childNode in list(rootNode.info.values())[0]:
        if(childNode.leaf):
            child = pydot.Node("val = " + childNode.attributeVal + "\n LEAF = " + childNode.label,
                               label="val = " + childNode.attributeVal + "\n LEAF = " + childNode.label)
            graph.add_node(child)
            edge = pydot.Edge("val = " + rootNode.attributeVal + "\n splitting on: " + list(rootNode.info.keys())[0],
                              "val = " + childNode.attributeVal + "\n LEAF = " + childNode.label)
        else:
            child = pydot.Node("val = " + childNode.attributeVal + "\n splitting on: " + list(childNode.info.keys())[0],
                               label="val = " + childNode.attributeVal + "\n splitting on: " + list(childNode.info.keys())[0])
            graph.add_node(child)
            edge = pydot.Edge("val = " + rootNode.attributeVal + "\n splitting on: " + list(rootNode.info.keys())[0],
                              "val = " + childNode.attributeVal + "\n splitting on: " + list(childNode.info.keys())[0])
        graph.add_edge(edge)
    G = nx.drawing.nx_pydot.from_pydot(graph)
    nx.draw(G)
    graph.graph_from_dot_file("example.dot")
            
        
    
    
    
    
    
    
    

    
    



        
    
    

    
    
    