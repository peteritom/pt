# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 14:37:32 2015

"""
import numpy as np
import cv2
#import cv
import os
import mpeg7_tools as mp

hist = np.load  ( 'edh_hist.npy' )
histQ = np.load ( 'edh_histQ.npy' )
D = mp.distEDHHist (hist, histQ)
sortedInd = np.argsort(D, axis=0)
q = 5
sortedInd = sortedInd[0:3,q]
qImgDir = "./project\\MPEG_7_descriptor\\edh_imgQ"
for (dirname, dirnames, filenames) in os.walk(qImgDir):
    qImg = cv2.resize(cv2.imread(os.path.join(dirname, filenames[q])), (100,100))
cv2.imshow("Query Image", qImg.astype('uint8'));

img = []
img.append(qImg)
l = 0
imgDir = "./project\\MPEG_7_descriptor\\edh_img"
for (dirname, dirnames, filenames) in os.walk(imgDir):
    for filename in filenames:
        for i in range(0,3):
            if l == sortedInd[i]:
                img.append(cv2.resize(cv2.imread(os.path.join(dirname, filename)), (100,100)))
        l = l+1

vis = np.concatenate((img[:]), axis=1)
cv2.imwrite("%d_query.jpeg" % q, vis.astype('uint8'))
cv2.imshow("Database Image", vis.astype('uint8'));
cv2.waitKey(0)