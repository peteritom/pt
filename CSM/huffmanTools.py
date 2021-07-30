# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 00:07:21 2015

"""

import Queue # for the Huffman-tree

class HuffmanNode(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right
        
def createTree(tupleList):  # expects (weight, value) pairs
    p = Queue.PriorityQueue()
    for i in tupleList:    # 1. Create a leaf node for each symbol
#        print i
        p.put(i)             #    and add it to the priority queue
    while p.qsize() > 1:         # 2. While there is more than one node
        l, r = p.get(), p.get()  # 2a. remove two highest nodes    
        node = HuffmanNode(l, r) # 2b. create internal node with children
        p.put((l[0]+r[0], node)) # 2c. add new node to queue      
    return p.get()               # 3. tree is complete - return root node (only 1 Node is in the PriorityQueue, so get() returns that one)
    
def walkTree(node, prefix="", code={}):
    weight, nodeOrValue = node    # n is either another HuffmanNode, or the greyscale value of a leave
#    print weight    
    if not isinstance(nodeOrValue, HuffmanNode):
        code[nodeOrValue] = prefix    # if it's a leave, Value will be the index of a dictionary, prefix is the belonging code
    else:
        walkTree(nodeOrValue.left, prefix + "0")
        walkTree(nodeOrValue.right, prefix + "1")
    return(code)