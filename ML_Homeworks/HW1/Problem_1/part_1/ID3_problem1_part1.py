from inspect import Attribute
from re import L
from Node_problem1_part1 import Node
from Function_Library_problem1_part1 import *

#contains the attributes of the set and their values
ATTRIBUTES = {}

DATA = pd.DataFrame

DEPTHS = []

    
#labels are the possible outputs of the dataset
def ID3(DataFrame, Attributes=None, depth=0, valOfNode=None):
    SubsetDict = DataFrame.to_dict()
    
    if(Attributes == None): Attributes = GetAttributesLeft(SubsetDict)#need to define the attributes for the first iteration
    
    labelkey = list(SubsetDict)[-1]
    labelvals = np.array(DataFrame[labelkey])
    data = DataFrame.values
    
    #if all of the labels are the same, return the common label.
    LabelsAreEqual = All_Labels_Are_Da_Same(DataFrame)
    if(LabelsAreEqual[0]):
        depth+=1
        leafnode = Node(depth=depth, leaf=True, label=LabelsAreEqual[1])
        DEPTHS.append(depth)
        return leafnode
    
    #if there are no attributes left to split on, return a leaf node with the most common label.
    elif(len(Attributes) == 0):
        depth+=1
        McL = MostCommonLabel(labelvals)[0]
        leafnode = Node(depth=depth, leaf=True, label=McL)
        DEPTHS.append(depth)
        return leafnode
    
    else:
        depth+=1
        #find the best attribute to split on
        AttributeToSplit = AttributeWithHighestInfoGain_Entropy(data,Attributes)
        print(AttributeToSplit)
        info = {AttributeToSplit:[]}
        #attributeVal is the value of the attribute that we decided to split on in the last node
        rootNode = Node(info=info, depth=depth, attributeVal=valOfNode)
        
        #Always add a leaf node to each root node that contains the most common label of the rootNode's subset.
        McL_subset = MostCommonLabel(labelvals)[0]
        mostCommonLabel_leafNode = Node(depth=depth+1, leaf=True, attributeVal="MCL", label=McL_subset)
        rootNode.info[AttributeToSplit].append(mostCommonLabel_leafNode)

        PossibleValsOfAttributeToSplit = GetValuesPossibleOfAttribute(DataFrame, AttributeToSplit)
        
        for val in PossibleValsOfAttributeToSplit:
            new_df = DataFrame
            new_df = SplitData(DataFrame, AttributeToSplit, val)
            #if the new subset is empty, add a leaf node with the most common label in the whole dataset as the leaf label.
            if(new_df.size == 0):
                McL = MostCommonLabel(labelvals)[0]
                leafNode = Node(depth=depth+1, leaf=True, attributeVal=val, label=McL)
                rootNode.info[AttributeToSplit].append(leafNode)
                DEPTHS.append(depth)

            #if the whole subset has only 1 label, just add a leaf node with that label.
            elif(len(np.unique(np.array(new_df[labelkey]))) == 1):
                leafNode = Node(depth=depth+1, leaf=True, attributeVal=val, label=list(new_df[labelkey])[0])
                rootNode.info[AttributeToSplit].append(leafNode)
                DEPTHS.append(depth)

            #otherwise, go ahead and run ID3 again, which will create another root node with the val
            # as the attributevalthat has other nodes below it.
            else:
                newAttributes = list(Attributes)
                newAttributes.remove(AttributeToSplit)
                rootNode.info[AttributeToSplit].append(ID3(new_df, Attributes=newAttributes, depth=depth, valOfNode=val))
        
        return rootNode
                
                
#problem1 binary dataset
filename = "/Users/jakehirst/Desktop/Machine Learning/ML_Homeworks/HW1/Problem_1/part_1/problem1_data_allthesamelabel.csv"

DATA = Read_Data(filename)
rootNode = ID3(DATA)
print("max depth of tree = " + str(max(DEPTHS)))
print("done!")

#problem1 binary dataset
TestFileName = "/Users/jakehirst/Desktop/Machine Learning/ML_Homeworks/HW1/Problem_1/part_1/problem1_data_allthesamelabel.csv"

columnTitles = DATA.columns.values 
CheckTreeAgainstTestData(TestFileName, rootNode, columnTitles)

