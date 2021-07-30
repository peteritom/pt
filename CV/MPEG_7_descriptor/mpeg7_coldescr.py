# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:10:29 2015

"""

import numpy as np
import cv2
#import cv
import os
import mpeg7_tools as mp

img = np.zeros((20,20))

hist = np.load  ( 'hist.npy'  )
histQ = np.load ( 'histQ.npy' )
N = np.size(hist,1) # number of database images
M = np.size(histQ,1) # number of query images
D = mp.distanceMatrix(hist, histQ)
sortedInd = np.argsort(D, axis=0)

q = 2
sortedInd = sortedInd[0:3,q]
qImgDir = "./project\\MPEG_7_descriptor\\queryImages"
#qImgDir  = "./project\\MPEG_7_descriptor\\edh_imgQ"

for (dirname, dirnames, filenames) in os.walk(qImgDir):
    qImg = cv2.imread(os.path.join(qImgDir, filenames[q]))

l = 0
img = []
img.append(cv2.resize(qImg,(100,100)))
imgDir = "./project\\MPEG_7_descriptor\\pictures"
#imgDir = "./project\\MPEG_7_descriptor\\edh_img"
for (dirname, dirnames, filenames) in os.walk(imgDir):
    for filename in filenames:
        for i in range(0,3):
            if l == sortedInd[i]:
                img.append(cv2.resize(cv2.imread(os.path.join(dirname, filename)), (100,100)))
        l = l+1
vis = np.concatenate((img[:]), axis=1)
cv2.imwrite("%d_queryc.jpeg" % q, vis.astype('uint8'))
cv2.imshow("Database Image", vis.astype('uint8'));
#cv2.imshow("win", img.astype('uint8'));
cv2.waitKey(0)