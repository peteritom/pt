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
print type(hist)
print type(histQ)
D = mp.distEDHHist (hist, histQ)
sortedInd = np.argsort(D, axis=0)
q = 0
sortedInd = sortedInd[0:3,q]
qImgDir = "C:\\Users\\Tomi\\Desktop\\ERASMUS_ISEL\\FELEV_KOZBEN\\ArtVision\\project\\MPEG_7_descriptor\\edh_imgQ"
for (dirname, dirnames, filenames) in os.walk(qImgDir):
    qImg = mp.imgResize(qImgDir, filenames[q])
cv2.imshow("Query Image", qImg.astype('uint8'));

l = 0
imgDir = "C:\\Users\\Tomi\\Desktop\\ERASMUS_ISEL\\FELEV_KOZBEN\\ArtVision\\project\\MPEG_7_descriptor\\edh_img"
for (dirname, dirnames, filenames) in os.walk(imgDir):
    for filename in filenames:
        for i in range(0,3):
            if l == sortedInd[i]:
                img = mp.imgResize(dirname, filename)
                cv2.imshow("%d. database Image" % l, img.astype('uint8'));
        l = l+1
cv2.waitKey(0)