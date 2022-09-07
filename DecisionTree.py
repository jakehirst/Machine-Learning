from cmath import log
from math import log2


inputs = [[0,0,1,0],[0,1,0,0],[0,0,1,1],[1,0,0,1],[0,1,1,0],[1,1,0,0],[0,1,0,1]]
outputs = [0,0,1,1,0,0,0]

#finds the entropy of a subset
def Entropy(Subset):
    total_plus = 0
    total_minus = 0
    for output in Subset:
        if(output == 1): total_plus += 1
        else: total_minus += 1
        
    total = total_minus + total_plus
    p_plus = total_plus / total
    p_minus = total_minus / total
    #not sure if this is supposed to be log_2
    entropy = -(p_plus)*log2(p_plus) - p_minus*log2(p_minus)
    return entropy

def information_gain(Subset, Attribute):
    return 0

    
    
    