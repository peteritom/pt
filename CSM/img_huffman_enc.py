# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 17:32:04 2015

"""

import cv2
import numpy as np

from operator import itemgetter
import huffmanTools


# reading image, converting it to greyscale
img = cv2.imread('./lena.tiff')
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('TestImageWindow',1)
cv2.imshow('TestImageWindow', grey)

# converting 2D array to 1D array
grey = np.reshape(grey, (1,np.size(grey)))

# making the histogram
hist = np.zeros((1,256))

for i in grey[0,:]: # iterate through all pixels of the image
    hist[(0, grey[0,i])] += 1
    
if np.sum(hist) != np.size(grey):
    print 'Histogram is faulty!'
else:
    print 'Histogram seems OK. (sum == pixel num of the picture)'

## numpy histogram, nemtom h mukodik pontosan, opencv-s jobb   
#hist2 = np.histogram(grey)
##if hist != hist2:
##    print "ERROR"
    
# openCV hist
hist3 = cv2.calcHist([grey],[0],None,[256],[0,256])
hist3_test = np.reshape(hist3, (1,256))
if hist.all() != hist3_test.all():
    print "ERROR"
else:
    print "sajat histo szamitas es az opencv-s megegyezik"


# HUFFMAN
# making a list of tuples of the histogram - (value, probability)
tupleList = list()
value = 0
for weight in hist[0,:]:
    if weight != 0.0:  # only save to list if probability is non-zero
        tupleList.append((weight, value))
    value += 1

# making the tree
root = huffmanTools.createTree(tupleList)

# walking the tree, assigning codes (codes is a dictionary with value-code pairs)
code = huffmanTools.walkTree(root)

# print
for i in sorted(tupleList, reverse=True):
    print i[1], '{:6.2f}'.format(i[0]), code[i[1]]

    

cv2.waitKey(0)

#def side_by_side(a, b, w):
#    a = a.split("\n")
#    b = b.split("\n")
#    n1 = len(a)
#    n2 = len(b)
#    if n1 < n2:
#        a.extend([" "*len(a[0])]*(n2-n1))
#    else:
#        b.extend([" "*len(b[0])]*(n1-n2))
#    r = [" "*len(a[0]) + "   ^   " + " "*len(b[0])]
#    r += ["/" + "-"*(len(a[0])-1) + "%7.3f" % w + "-"*(len(b[0])-1) + "\\"]
#    for l1, l2 in zip(a, b):
#        r.append(l1 + "       " + l2)
#    return "\n".join(r)
# 
#def print_tree(node):
#    w, n = node
#    if not isinstance(n, HuffmanNode):
#        return "%s = %.3f" % (n, w)
#    else:
#        l = print_tree(n.left)
#        r = print_tree(n.right)
#        return side_by_side(l, r, w)
# 
#print print_tree(node)
 