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

ATTRIBUTE_VALUES = {0: [0,1],
                    1: [0,1],
                    2: [0,1],
                    3: [0,1]}

for row in DATA:
    POSSIBLE_OUTPUTS.add(row[4])

TOTAL_ENTROPY = float()

def main():
    TOTAL_ENTROPY = Entropy(DATA)
    print(Information_gain(DATA, 0))
    


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
     
     
     
                        
if __name__ == "__main__":
    main()                        

            
                

    
    
    