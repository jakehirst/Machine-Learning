from cProfile import label
from cmath import log
from math import log2


DATA = [[0,0,1,0,0],
        [0,1,0,0,0],
        [0,0,1,1,1],
        [1,0,0,1,1],
        [0,1,1,0,0],
        [1,1,0,0,0],
        [0,1,0,1,0]]

DATA1 = [[0,0,1,0,0],
        [0,1,0,0,0],
        [0,0,1,1,0],
        [1,0,0,1,0],
        [0,1,1,0,0],
        [1,1,0,0,0],
        [0,1,0,1,0]]


POSSIBLE_OUTPUTS = set()

ATTRIBUTE_VALUES = {0: [0,1],
                    1: [0,1],
                    2: [0,1],
                    3: [0,1]}

for row in DATA:
    POSSIBLE_OUTPUTS.add(row[4])

TOTAL_ENTROPY = float()
    

#finds the entropy of a subset
def Entropy(Subset):
    i = 0
    p = []
    thingstosum = []
    whole = len(Subset)
    for output in POSSIBLE_OUTPUTS:
        part = 0
        for row in Subset:
            if(row[4] == output): part += 1
        p.append(part/whole)
        if(p[i] == 0 or p[i] == 1):
            return 0 #without this, log_2(0) or log_2(1) is NaN
        thingstosum.append(p[i]* log2(p[i]))
        i+=1

    entropy = -sum(thingstosum)
    return entropy

#gets the information gain of a single attribute based on the entropy
#Attribute in this case is an integer from 0-3, referring to x_1 - x_3.
def Information_gain(BigSubset, Attribute):
    thingstosum = []
    for value in ATTRIBUTE_VALUES[Attribute]:
        subset = []
        for row in BigSubset:
            if(row[Attribute] == value):
                subset.append(row)
        thingstosum.append((len(subset)/len(BigSubset)) * Entropy(subset))
    return Entropy(DATA) - sum(thingstosum)
     
     
class Node():
    #subset is self explanatory (array)
    #attributes is self explanatory (array of attributes left to go through)
    #parent is the node above this node in the tree (Node)
    #leaf is whether it is a leaf node or not (boolean)
    def __init__(self, subset, attributes, parentNode, leaf):
        self.subset = subset
        self.attributes = attributes
        self.parentNode = parentNode
        self.leaf = leaf
        
def All_Labels_Are_Da_Same(Subset):
    label_index = len(Subset[0]) - 1 #location of the label within any row of the subset
    commonlabel = Subset[1][label_index]
    for row in Subset:
        if(row[label_index] != commonlabel):
            return False, None
    return True, commonlabel


def MostCommonLabel(Subset):
    label_index = len(Subset[0]) - 1 #location of the label within any row of the subset
    labels_and_frequency = {}
    for row in Subset:
        if(labels_and_frequency.keys().__contains__(row[label_index])):
            labels_and_frequency[row[label_index]] += 1
        else:
            labels_and_frequency[row[label_index]] = 1

    Max = max(labels_and_frequency.values())
    
    mostCommonLabels = list(labels_and_frequency.keys())[list(labels_and_frequency.values()).index(100)]
    return mostCommonLabels[0]


def ID3_Starter(DataSet):
    attributes = []
    for key in ATTRIBUTE_VALUES.keys(): attributes.append(key)
    node = Node(DataSet, attributes, None, None)
    ID3(node, node.subset, node.attributes)
    
    
#labels are the possible outputs of the dataset
def ID3(CurrentNode, Subset, Attributes):
    ans = All_Labels_Are_Da_Same(Subset)
    if(ans[0] == True):
        print('all labels are the same')
        return Node([],[], CurrentNode, ans[1])
    print('labels are not all the same yet')
    if(len(Attributes) == 0):
        label = MostCommonLabel(Subset)
    return Node(Subset, Attributes, CurrentNode, label)

        
ID3_Starter(DATA1)

        
    
    

    
    
    