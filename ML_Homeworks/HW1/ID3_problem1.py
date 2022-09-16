from inspect import Attribute
from re import L
from Node import Node
from Function_Library import *

#contains the attributes of the set and their values
ATTRIBUTES = {}

DATA = pd.DataFrame

    
#labels are the possible outputs of the dataset
def ID3(DataFrame, Attributes=None, depth=0):
    SubsetDict = DataFrame.to_dict()
    
    if(Attributes == None): Attributes = GetAttributesLeft(SubsetDict)#need to define the attributes for the first iteration
    
    labelkey = list(SubsetDict)[-1]
    labelvals = np.array(DataFrame[labelkey])
    data = DataFrame.values
    
    #if all of the labels are the same, return the common label.
    LabelsAreEqual = All_Labels_Are_Da_Same(DataFrame)
    if(LabelsAreEqual[0]):
        
        return Node(leaf=True, label=LabelsAreEqual[1])
    
    #if there are no attributes left to split on, return a leaf node with the most common label.
    elif(len(Attributes) == 0):
        McL = MostCommonLabel(labelvals)[0]
        return Node(leaf=True, label=McL)
    
    else:
        depth+=1
        #find the best attribute to split on
        AttributeToSplit = AttributeWithHighestInfoGain(data,Attributes)
        print(AttributeToSplit)
        info = {AttributeToSplit:[]}
        rootNode = Node(info=info, depth=depth)
        PossibleValsOfAttributeToSplit = GetValuesPossibleOfAttribute(DataFrame, AttributeToSplit)
        
        for val in PossibleValsOfAttributeToSplit:
            new_df = DataFrame
            new_df = SplitData(DataFrame, AttributeToSplit, val)
            if(new_df.size == 0):
                McL = MostCommonLabel(labelvals)[0]
                leafNode = Node(depth=depth+1, leaf=True, label=McL)
                rootNode.info[AttributeToSplit].append(leafNode)
            elif(len(np.unique(np.array(new_df[labelkey]))) == 1):
                leafNode = Node(depth=depth+1, leaf=True, label=list(new_df[labelkey])[0])
                rootNode.info[AttributeToSplit].append(leafNode)
            else:
                newAttributes = list(Attributes)
                newAttributes.remove(AttributeToSplit)
                rootNode.info[AttributeToSplit].append(ID3(new_df, Attributes=newAttributes, depth=depth))
        
        return rootNode
                
                
                
            
            
        

        
        
    
        
#filename = "/Users/jakehirst/Desktop/Machine Learning/ML_Homeworks/HW1/problem1_data_allthesamelabel.csv"
filename = "/Users/jakehirst/Desktop/Machine Learning/ML_Homeworks/HW1/problem1_data.csv"
DATA = Read_Data(filename)
rootNode = ID3(DATA)
print("done!")
