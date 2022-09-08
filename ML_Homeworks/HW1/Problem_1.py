from cmath import log
from math import log2


DATA = [[0,0,1,0,0],
        [0,1,0,0,0],
        [0,0,1,1,1],
        [1,0,0,1,1],
        [0,1,1,0,0],
        [1,1,0,0,0],
        [0,1,0,1,0]]

POSSIBLE_OUTPUTS = set()

ATTRIBUTE_VALUES = {int, set()}

for row in DATA:
    POSSIBLE_OUTPUTS.add(row[4])

TOTAL_ENTROPY = float()

def main():
    TOTAL_ENTROPY = Entropy(DATA)
    

def Find_attribute_Values():
    for row in DATA:
        for i in range(0,len(DATA[1]) - 1):
            if(not ATTRIBUTE_VALUES[i] == row[i]):
                ATTRIBUTE_VALUES[i].add(row[i])


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
        thingstosum.append(p[i]* log2(p[i]))
        i+=1

    entropy = -sum(thingstosum)
    return entropy

#gets the information gain of a single attribute based on the entropy
#Attribute in this case is an integer from 0-3, referring to x_1 - x_3.
def Information_gain(Attribute):
    for value in Attribute_values[Attribute]:
        subset = []
        for row in DATA:
            if(row[Attribute] == value and row[1] == output):
                count += 1 
                        
    return TOTAL_ENTROPY - sum(thingstosum)
     
     
     
                        
if __name__ == "__main__":
    main()                        

            
                

    
    
    