class Node():
    
    #info is a dictionary, with the key being the attribute that the node splits on, and the value being the list of nodes
    #directly below it, which could either be leaf nodes with just their label, or it could be another node with more nodes attatched.
    
    #if it is a leaf node, leaf is true and value is the value of the leaf node
    #the depth is the depth of the node 
    def __init__(self, branch=None, info=None, depth=0, leaf=False, label=None):
        self.info = info
        self.depth = depth
        self.leaf = leaf
        self.label = label
        
        