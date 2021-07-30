# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:33:38 2015

"""

import cv2
import numpy as np
import os
import tools

origFileSizeList = []
jpegFileSizeList = []

# saving original frames to JPEGs
frameDir = 'D:\\DOCUMENTS\\SULI\\ISEL\\CSM\\workspace\\video\\frames'
frameList = []
jpegFileNameList = []
for (dirname, dirnames, filenames) in os.walk(frameDir):
    for filename in filenames:
        img = cv2.imread(os.path.join(dirname, filename))
        size = os.path.getsize(os.path.join(dirname, filename))
        origFileSizeList.append(float(size))
        frameList.append(img) 
        name, extension = filename.split('.')
        cv2.imwrite('%s.jpeg' %name, img, (cv2.cv.CV_IMWRITE_JPEG_QUALITY, 75)) 
        jpegFileNameList.append('%s.jpeg' %name)
      

# calculate entropy, energy of the original frames
frameEntropyList = []
frameEnergyList = []
for frame in frameList:
    entropy = tools.computeEntropy(frame)
    frameEntropyList.append(entropy)
    energy = tools.computeEnergy(frame)
    frameEnergyList.append(energy)
    
# reading in the JPEGs; and calculate the entropy, energy of them
frameJPEGList = []
frameJPEGEntropyList = []
frameJPEGEnergyList = []
for file in jpegFileNameList:
    jpegFrame = cv2.imread(file)
    size = os.path.getsize(file)
    jpegFileSizeList.append(float(size))
    frameJPEGList.append(jpegFrame)
    entropy = tools.computeEntropy(jpegFrame)
    frameJPEGEntropyList.append(entropy)
    energy = tools.computeEnergy(jpegFrame)
    frameJPEGEnergyList.append(energy)
    
# calculating SNR of the original and the compressed-decompressed frames
SNRList = []
for orig, jpeg in zip(frameList, frameJPEGList):
    snr = tools.computeSNR(orig, jpeg)
    SNRList.append(snr)
    
# calculating compression ratio
compRatioList = []
for origSize, jpegSize in zip(origFileSizeList, jpegFileSizeList):
    compRatio = origSize/jpegSize
    compRatioList.append(compRatio)

        
cv2.imshow('win1', img)
cv2.waitKey(0)

