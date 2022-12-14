from inspect import Attribute
from re import L
from .Node import Node
from .Function_Library import *
from IPython.display import Image, display
from multiprocessing import Pool
import math as m
import random

#contains the attributes of the set and their values


class ID3():
    
    def __init__(self):
        self.DEPTHS = []
    #labels are the possible outputs of the dataset
    def ID3(self, DataFrame, InfoGainMethod, Attributes=None, depth=0, valOfNode=None, MaxDepth=None):
        SubsetDict = DataFrame.to_dict()
        
        
        if(Attributes == None): Attributes = GetAttributesLeft(SubsetDict)#need to define the attributes for the first iteration
        
        labelkey = list(SubsetDict)[-2]
        labelvals = np.array(DataFrame[labelkey])
        data = DataFrame.values
        
        #if all of the labels are the same, return the common label.
        LabelsAreEqual = All_Labels_Are_Da_Same(labelvals)
        if(LabelsAreEqual[0]):
            depth+=1
            leafnode = Node(depth=depth, leaf=True, label=LabelsAreEqual[1])
            self.DEPTHS.append(depth)
            return leafnode
        
        #if there are no attributes left to split on, return a leaf node with the most common label.
        elif(len(Attributes) == 0):
            depth+=1
            McL = MostCommonLabel(DataFrame,labelvals)[0]
            leafnode = Node(depth=depth, leaf=True, label=McL)
            self.DEPTHS.append(depth)
            return leafnode
        
        else:
            
            #find the best attribute to split on
            if(InfoGainMethod == "MajorityError"):
                poss_labels = list(np.unique(data[:,data.shape[1] - 2]))
                AttributeToSplit = AttributeWithHighestInfoGain_MajorityError(data,Attributes,poss_labels)
            elif(InfoGainMethod == "GiniIndex"):
                AttributeToSplit = AttributeWithHighestInfoGain_GiniIndex(data,Attributes)
            elif(InfoGainMethod == "Entropy"):
                AttributeToSplit = AttributeWithHighestInfoGain_Entropy(data,Attributes)
            else:
                print("Need to specify the InfoGainMethod")
            
            #print(AttributeToSplit)
            info = {AttributeToSplit:[]}
            if(valOfNode == None):
                valOfNode = "Root"
            #attributeVal is the value of the attribute that we decided to split on in the last node
            rootNode = Node(info=info, depth=depth, attributeVal=valOfNode)
            
            McL_subset = MostCommonLabel(DataFrame,labelvals)[0]
            mostCommonLabel_leafNode = Node(depth=depth+1, leaf=True, attributeVal=valOfNode, label=McL_subset)
            #print(f"depth = {depth}")
            if(depth == MaxDepth):
                self.DEPTHS.append(depth)
                return mostCommonLabel_leafNode
            
            #Always add a leaf node to each root node that contains the most common label of the rootNode's subset.
            rootNode.info[AttributeToSplit].append(mostCommonLabel_leafNode)
            
            depth+=1
            PossibleValsOfAttributeToSplit = GetValuesPossibleOfAttribute(DataFrame, AttributeToSplit)
            
            #go thru each possible value of the attribute youre splitting on
            for val in PossibleValsOfAttributeToSplit:
                
                #print("Splitting attribute : " + str(AttributeToSplit))
                #print("Attribute value : " + str(val))
                new_df = DataFrame
                new_df = SplitData(DataFrame, AttributeToSplit, val)
                #print(new_df.to_latex())
                #if the new subset is empty, add a leaf node with the most common label in the whole dataset as the leaf label.
                if(new_df.size == 0):
                    #print("subset is empty")
                    McL = MostCommonLabel(DataFrame, labelvals)[0]
                    leafNode = Node(depth=depth+1, leaf=True, attributeVal=val, label=McL)
                    rootNode.info[AttributeToSplit].append(leafNode)
                    self.DEPTHS.append(depth)

                #if the whole subset has only 1 label, just add a leaf node with that label.
                elif(len(np.unique(np.array(new_df[labelkey]))) == 1):
                    leafNode = Node(depth=depth+1, leaf=True, attributeVal=val, label=list(new_df[labelkey])[0])
                    rootNode.info[AttributeToSplit].append(leafNode)
                    self.DEPTHS.append(depth)

                #otherwise, go ahead and run ID3 again, which will create another root node with the val
                # as the attributevalthat has other nodes below it.
                else:
                    newAttributes = list(Attributes)
                    newAttributes.remove(AttributeToSplit)
                    rootNode.info[AttributeToSplit].append(self.ID3(new_df, InfoGainMethod, Attributes=newAttributes, depth=depth, valOfNode=val, MaxDepth=MaxDepth))
            
            return rootNode
              


    """ runs ID3 algorithm, but this time with a more random selection of attributes. Used for random forests. """
    def randomID3(self, DataFrame, InfoGainMethod, num_random_attributes=2 ,Attributes=None, depth=0, valOfNode=None, MaxDepth=None):
        SubsetDict = DataFrame.to_dict()
        
        
        if(Attributes == None): Attributes = GetAttributesLeft(SubsetDict)#need to define the attributes for the first iteration
        
        labelkey = list(SubsetDict)[-2]
        labelvals = np.array(DataFrame[labelkey])
        data = DataFrame.values
        
        #if all of the labels are the same, return the common label.
        LabelsAreEqual = All_Labels_Are_Da_Same(labelvals)
        if(LabelsAreEqual[0]):
            depth+=1
            leafnode = Node(depth=depth, leaf=True, label=LabelsAreEqual[1])
            self.DEPTHS.append(depth)
            return leafnode
        
        #if there are no attributes left to split on, return a leaf node with the most common label.
        elif(len(Attributes) == 0):
            depth+=1
            McL = MostCommonLabel(DataFrame,labelvals)[0]
            leafnode = Node(depth=depth, leaf=True, label=McL)
            self.DEPTHS.append(depth)
            return leafnode
        
        else:
            #COMMENT change the new_length_attributes from 2 to 4 to 6
            #new_length_attributes = m.ceil(len(Attributes)/2)
            if(len(Attributes) > num_random_attributes):
                tempAttributes = random.sample(Attributes, num_random_attributes)
            else:
                tempAttributes = Attributes
            #find the best attribute to split on
            if(InfoGainMethod == "MajorityError"):
                poss_labels = list(np.unique(data[:,data.shape[1] - 2]))
                AttributeToSplit = AttributeWithHighestInfoGain_MajorityError(data,tempAttributes,poss_labels)
            elif(InfoGainMethod == "GiniIndex"):
                AttributeToSplit = AttributeWithHighestInfoGain_GiniIndex(data,tempAttributes)
            elif(InfoGainMethod == "Entropy"):
                AttributeToSplit = AttributeWithHighestInfoGain_Entropy(data,tempAttributes)
            else:
                print("Need to specify the InfoGainMethod")
            
            #print(AttributeToSplit)
            info = {AttributeToSplit:[]}
            if(valOfNode == None):
                valOfNode = "Root"
            #attributeVal is the value of the attribute that we decided to split on in the last node
            rootNode = Node(info=info, depth=depth, attributeVal=valOfNode)
            
            McL_subset = MostCommonLabel(DataFrame,labelvals)[0]
            mostCommonLabel_leafNode = Node(depth=depth+1, leaf=True, attributeVal=valOfNode, label=McL_subset)
            #print(f"depth = {depth}")
            if(depth == MaxDepth):
                self.DEPTHS.append(depth)
                return mostCommonLabel_leafNode
            
            #Always add a leaf node to each root node that contains the most common label of the rootNode's subset.
            rootNode.info[AttributeToSplit].append(mostCommonLabel_leafNode)
            
            depth+=1
            PossibleValsOfAttributeToSplit = GetValuesPossibleOfAttribute(DataFrame, AttributeToSplit)
            
            #go thru each possible value of the attribute youre splitting on
            for val in PossibleValsOfAttributeToSplit:
                
                #print("Splitting attribute : " + str(AttributeToSplit))
                #print("Attribute value : " + str(val))
                new_df = DataFrame
                new_df = SplitData(DataFrame, AttributeToSplit, val)
                #print(new_df.to_latex())
                #if the new subset is empty, add a leaf node with the most common label in the whole dataset as the leaf label.
                if(new_df.size == 0):
                    #print("subset is empty")
                    McL = MostCommonLabel(DataFrame, labelvals)[0]
                    leafNode = Node(depth=depth+1, leaf=True, attributeVal=val, label=McL)
                    rootNode.info[AttributeToSplit].append(leafNode)
                    self.DEPTHS.append(depth)

                #if the whole subset has only 1 label, just add a leaf node with that label.
                elif(len(np.unique(np.array(new_df[labelkey]))) == 1):
                    leafNode = Node(depth=depth+1, leaf=True, attributeVal=val, label=list(new_df[labelkey])[0])
                    rootNode.info[AttributeToSplit].append(leafNode)
                    self.DEPTHS.append(depth)

                #otherwise, go ahead and run ID3 again, which will create another root node with the val
                # as the attributevalthat has other nodes below it.
                else:
                    newAttributes = list(Attributes)
                    newAttributes.remove(AttributeToSplit)
                    rootNode.info[AttributeToSplit].append(self.randomID3(new_df, InfoGainMethod, Attributes=newAttributes, depth=depth, valOfNode=val, MaxDepth=MaxDepth))
            
            return rootNode
        

                    
    """ This function runs the ID3 algorithm and tests it against the Testdf dataset, printing the results. """
    def runID3(self, data, InfoGainMethod, Maxdepth, Testdf):
        rootNode = self.ID3(data, InfoGainMethod, MaxDepth=Maxdepth)
        columnTitles = data.columns.values 
        error = CheckTreeAgainstTestData(Testdf, rootNode, columnTitles)
        return [rootNode, error, Maxdepth]


    '''preps the data with FillMissingAttributes and binarize_numeric_vals'''
    def prepData(self, filename, MissingIndicator=None, howToFill=None, columns_to_binarize=None):
        data = Read_Data(filename)
        #if there is a missing indicator and a howToFill method, then go ahead and fill the missing attributes.
        if((not MissingIndicator == None) and (not howToFill == None)):
            data = FillMissingAttributes(data, MissingIndicator, howToFill)
        #if there are columns to binarize, then go ahead and binarize those columns
        if(not columns_to_binarize == None):
            data = binarize_numeric_vals(data, columns_to_binarize)
        return data


    '''uses multiprocessing to prep the data quickly'''
    def prepData_quickly(self, filenames, MissingIndicator=None, howToFill=None, columns_to_binarize=None):
        prep_pool = Pool(2) #change cores to decide how many cores to use in multiprocessing 
        prepped_data = []
        for filename in filenames:
            rootNode_and_Error = prep_pool.apply_async(self.prepData, [filename, MissingIndicator, howToFill, columns_to_binarize])
            prepped_data.append(rootNode_and_Error)
        for data in prepped_data:
            data.wait()
        prep_pool.close()
        prep_pool.join()
        
        data = prepped_data[0].get()
        Testdf = prepped_data[1].get()
        return data, Testdf


if __name__ == "__main__":
    obj = ID3()
    """Training data File Names"""
    #problem2 tennis dataset
    #filename = "/Users/jakehirst/Desktop/Machine_Learning/DecisionTree/TennisDataset.csv"
    #problem2 tennis dataset with missing attribute
    #filename = "/Users/jakehirst/Desktop/Machine_Learning/DecisionTree/TennisDataset_with_missing_attribute.csv"  
    #bank training dataset
    filename = "/Users/jakehirst/Desktop/Machine_Learning/DecisionTree/bank/train.csv"
    #car training dataset
    #filename = "/Users/jakehirst/Desktop/Machine_Learning/DecisionTree/car/train.csv"

    """Test data file Names"""
    #problem2 tennis dataset
    #TestFileName = "/Users/jakehirst/Desktop/Machine_Learning/DecisionTree/TennisDataset.csv"
    #problem2 tennis dataset with missing attribute
    #TestFileName = "/Users/jakehirst/Desktop/Machine_Learning/DecisionTree/TennisDataset_with_missing_attribute.csv"
    #bank tester dataset
    TestFileName = "/Users/jakehirst/Desktop/Machine_Learning/DecisionTree/bank/test.csv"
    #bank training dataset
    #TestFileName = "/Users/jakehirst/Desktop/Machine_Learning/DecisionTree/bank/train.csv"
    #car training dataset
    #TestFileName = "/Users/jakehirst/Desktop/Machine_Learning/DecisionTree/car/train.csv"
    #car test dataset
    #TestFileName = "/Users/jakehirst/Desktop/Machine_Learning/DecisionTree/car/test.csv"


    """ --------  RUNNING CODE HERE  -------- """

    """ DATA PREPROCESSING """
    columns_to_binarize = ["age", "balance","day","duration","campaign","pdays", "previous"]

    # data = Read_Data(filename)
    # data = FillMissingAttributes(data, 'unknown', 'b')
    # data = binarize_numeric_vals(data, columns_to_binarize)

    # Testdf = Read_Data(TestFileName)
    # Testdf = FillMissingAttributes(Testdf, 'unknown', 'b')
    # Testdf = binarize_numeric_vals(Testdf, columns_to_binarize)
    """ DATA PREPROCESSING """

    filenames = [filename, TestFileName]
    data, Testdf = obj.prepData_quickly(filenames, 'unknown', 'a', columns_to_binarize)

    POSS_LABELS = list(np.unique(np.array(data[data.columns[len(data.columns)-2]])))

    rootNode = obj.randomID3(data, "GiniIndex")
    
    x = obj.runID3(data, "GiniIndex", 5, Testdf)
    print(x)
    """ Running ID3 with multiple MaxDepths """
    InfoGainMethod = "GiniIndex" #can replace this with "MajorityError" or "Entropy" or "GiniIndex"
    p = Pool(8) #change this to decide how many cores to use in multiprocessing
    results = []
    for MaxDepth in range(17,0, -1):
        rootNode_and_Error = p.apply_async(obj.runID3, [data, InfoGainMethod, MaxDepth, Testdf])
        results.append(rootNode_and_Error)
    for r in results:
        r.wait()
    p.close()
    p.join()

    for r in results:
        print(r._value)
    print("done")

    """ Running ID3 with multiple MaxDepths """
    """ --------  RUNNING CODE HERE  -------- """
