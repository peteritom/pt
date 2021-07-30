# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:57:30 2015

"""

import cv2
import numpy as np
import os
import tools

origFrameList = []
origFileSizeList = []
jpegFileSizeList = []

# saving I frame as a JPEG
dirName = 'C:\\Users\\Tomi\\Desktop\\ERASMUS_ISEL\\FELEV_KOZBEN\\CSM_szuletes\\project3-video_danika\\frames'
IframeFileName = 'bola_a.tiff'
Iframe = cv2.imread(os.path.join(dirName, IframeFileName))
origFrameList.append(Iframe)
size = os.path.getsize(os.path.join(dirName, IframeFileName))
origFileSizeList.append(float(size))

outputFolder = 'C:\\Users\\Tomi\\Desktop\\ERASMUS_ISEL\\FELEV_KOZBEN\\CSM_szuletes\\project3-video_danika\\IandPJPEGs'
Iname, Iextension = IframeFileName.split('.')
cv2.imwrite('IandPJPEGs\\%s.jpeg' %Iname, Iframe, (cv2.cv.CV_IMWRITE_JPEG_QUALITY, 75)) 

# calculating P frames, saving them to JPEGs
IframeJPEG = cv2.imread(os.path.join(outputFolder, '%s.jpeg' %Iname)) # reading back the JPEG I frame for this

for (dirname, dirnames, filenames) in os.walk(dirName):
    for filename in filenames:
        if filename != IframeFileName:
            img = cv2.imread(os.path.join(dirname, filename))
            size = os.path.getsize(os.path.join(dirname, filename))
            origFileSizeList.append(float(size))
            origFrameList.append(img)
            Pframe = img - IframeJPEG
            Pframe += 128 # for the pixel values to be between 0...255
#            print Pframe
            name, extension = filename.split('.')
            cv2.imwrite('IandPJPEGs\\%s.jpeg' %name, Pframe, (cv2.cv.CV_IMWRITE_JPEG_QUALITY, 75))
            
# read back I and P frames, calculate energy, entropy and restore original frames
jpegEnergyList = []
jpegEntropyList = []  
restoredFrameList = []
for (dirname, dirnames, filenames) in os.walk(outputFolder):
    for filename in filenames:
        imgJPEG = cv2.imread(os.path.join(dirname, filename))
        size = os.path.getsize(os.path.join(dirname, filename))
        jpegFileSizeList.append(float(size))
        # calc energy
        energy = tools.computeEnergy(imgJPEG)
        jpegEnergyList.append(energy)
        # calc entropy
        entropy = tools.computeEntropy(imgJPEG)
        jpegEntropyList.append(entropy)
        # restore frame
        if filename == (Iname + '.jpeg'):
            restoredFrame = imgJPEG
        else:
            restoredFrame = imgJPEG + IframeJPEG
            restoredFrame -= 128
        cv2.imwrite('IandPJPEGs_restored\\%s.jpeg' %filename, restoredFrame)
        restoredFrameList.append(restoredFrame)

# calculating SNR and energy, entropy for the original frames
origEntropyList = []
origEnergyList = []
SNRList = []
for orig, jpeg in zip(origFrameList, restoredFrameList):
    snr = tools.computeSNR(orig, jpeg)
    SNRList.append(snr)
    
    energy = tools.computeEnergy(orig)
    origEnergyList.append(energy)
    
    entropy = tools.computeEntropy(orig)
    origEntropyList.append(entropy)
    
# calculating compression ratio
compRatioList = []
for origSize, jpegSize in zip(origFileSizeList, jpegFileSizeList):
    compRatio = origSize/jpegSize
    compRatioList.append(compRatio)
    
        


cv2.imshow('win1', Iframe)
cv2.waitKey(0)