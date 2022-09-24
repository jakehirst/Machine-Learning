from inspect import Attribute
from re import L
from Node import Node
from Function_Library import *
from IPython.display import Image, display

#contains the attributes of the set and their values
ATTRIBUTES = {}

DATA = pd.DataFrame

DEPTHS = []

POSS_LABELS = []

    
#labels are the possible outputs of the dataset
def ID3(DataFrame, Attributes=None, depth=0, valOfNode=None, MaxDepth=None):
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
        DEPTHS.append(depth)
        return leafnode
    
    #if there are no attributes left to split on, return a leaf node with the most common label.
    elif(len(Attributes) == 0):
        depth+=1
        McL = MostCommonLabel(DataFrame,labelvals)[0]
        leafnode = Node(depth=depth, leaf=True, label=McL)
        DEPTHS.append(depth)
        return leafnode
    
    else:
        
        #find the best attribute to split on
        #AttributeToSplit = AttributeWithHighestInfoGain_MajorityError(data,Attributes,POSS_LABELS)
        #AttributeToSplit = AttributeWithHighestInfoGain_GiniIndex(data,Attributes)
        AttributeToSplit = AttributeWithHighestInfoGain_Entropy(data,Attributes)
        
        #print(AttributeToSplit)
        info = {AttributeToSplit:[]}
        if(valOfNode == None):
            valOfNode = "Root"
        #attributeVal is the value of the attribute that we decided to split on in the last node
        rootNode = Node(info=info, depth=depth, attributeVal=valOfNode)
        
        #Always add a leaf node to each root node that contains the most common label of the rootNode's subset.
        McL_subset = MostCommonLabel(DataFrame,labelvals)[0]
        mostCommonLabel_leafNode = Node(depth=depth+1, leaf=True, attributeVal="MCL", label=McL_subset)
        if(depth == MaxDepth):
            DEPTHS.append(depth)
            return mostCommonLabel_leafNode
        rootNode.info[AttributeToSplit].append(mostCommonLabel_leafNode)
        
        depth+=1
        PossibleValsOfAttributeToSplit = GetValuesPossibleOfAttribute(DataFrame, AttributeToSplit)
        
        #go thru each possible value of the attribute youre splitting on
        for val in PossibleValsOfAttributeToSplit:
            
            print("Splitting attribute : " + AttributeToSplit)
            print("Attribute value : " + val)
            new_df = DataFrame
            new_df = SplitData(DataFrame, AttributeToSplit, val)
            print(new_df.to_latex())
            #if the new subset is empty, add a leaf node with the most common label in the whole dataset as the leaf label.
            if(new_df.size == 0):
                #print("subset is empty")
                McL = MostCommonLabel(DataFrame, labelvals)[0]
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
                rootNode.info[AttributeToSplit].append(ID3(new_df, Attributes=newAttributes, depth=depth, valOfNode=val, MaxDepth=MaxDepth))
        
        return rootNode
                
                
#problem2 tennis dataset
#filename = "/Users/jakehirst/Desktop/Machine Learning/DecisionTree/TennisDataset.csv"


#problem2 tennis dataset with missing attribute
filename = "/Users/jakehirst/Desktop/Machine Learning/DecisionTree/TennisDataset_with_missing_attribute.csv"
               
            
#bank training dataset
#filename = "/Users/jakehirst/Desktop/Machine Learning/DecisionTree/bank/train.csv"

#car training dataset
#filename = "/Users/jakehirst/Desktop/Machine Learning/DecisionTree/car/train.csv"

data = Read_Data(filename)
#DATA = FillMissingAttributes(data, 'unknown', 'a')
#columns_to_binarize = ["age", "balance","day","duration","campaign","pdays", "previous"]
#data = binarize_numeric_vals(data, columns_to_binarize)
DATA = FillMissingAttributes(data, 'Missing', 'c')
POSS_LABELS = list(np.unique(np.array(DATA[DATA.columns[len(DATA.columns)-2]])))

#problem2 tennis dataset
#TestFileName = "/Users/jakehirst/Desktop/Machine Learning/DecisionTree/TennisDataset.csv"

#problem2 tennis dataset with missing attribute
TestFileName = "/Users/jakehirst/Desktop/Machine Learning/DecisionTree/TennisDataset_with_missing_attribute.csv"

#bank tester dataset
#TestFileName = "/Users/jakehirst/Desktop/Machine Learning/DecisionTree/bank/test.csv"

#bank training dataset
#TestFileName = "/Users/jakehirst/Desktop/Machine Learning/DecisionTree/bank/train.csv"


#car training dataset
#TestFileName = "/Users/jakehirst/Desktop/Machine Learning/DecisionTree/car/train.csv"

#car test dataset
#TestFileName = "/Users/jakehirst/Desktop/Machine Learning/DecisionTree/car/test.csv"
Testdf = pd.read_csv(TestFileName)
# Testdf = binarize_numeric_vals(Testdf, columns_to_binarize)
# Testdf = FillMissingAttributes(Testdf, 'unknown', 'a')

rootNode = ID3(DATA, MaxDepth=None)
print("max depth of tree = " + str(max(DEPTHS)))


columnTitles = DATA.columns.values 
CheckTreeAgainstTestData(Testdf, rootNode, columnTitles)
for i in range(1,18):

    rootNode = ID3(DATA, MaxDepth=i)
    print("max depth of tree = " + str(max(DEPTHS)))


    columnTitles = DATA.columns.values 
    CheckTreeAgainstTestData(Testdf, rootNode, columnTitles)

