# -*- coding: utf-8 -*-
"""
Created on Mon Jun 01 16:48:40 2015

"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:57:30 2015

@author: Daniel
"""

import cv2
import numpy as np
import os
import tools

## Functions

def calcMotionVectors(refFrame, actualFrame): # returns the motionVectors for blocks and motionCorrected image
    bS = 16
    shape = np.shape(actualFrame)    
    width = shape[1]
    height = shape[0]
#    numOfBlocks = width/16 * height/16
    modImg = np.zeros(shape)
    motionVectors = np.zeros((height/bS, width/bS, 2))
    for y in range(0, height/bS):
        for x in range(0, width/bS):
            motionVector = findMostSimilarBlock(refFrame, actualFrame, shape, x*bS, y*bS, bS)
            motionVectors[y, x, 0] = motionVector[0]
            motionVectors[y, x, 1] = motionVector[1]
            modImg[y*bS:y*bS+bS, x*bS:x*bS+bS] = refFrame[y*bS+motionVector[0]:y*bS+motionVector[0]+bS, x*bS+motionVector[1]:x*bS+motionVector[1]+bS]         
            
    return motionVectors, modImg
        
def findMostSimilarBlock(refFrame, actualFrame, shape, actX, actY, bS): # bS = blockSize
    x_s = max(0, actX-bS-1) # x_start
    y_s = max(0, actY-bS-1)
    x_e = min(actX+bS+bS-1, shape[1]) # x_end
    y_e = min(actY+bS+bS-1, shape[0])
    actualBlock = actualFrame[actY:actY+bS, actX:actX+bS]
#    print 'ref', np.shape(refBlock)
#    print 'x_s', x_s
#    print 'x_e', x_e
    slidingBlock = refFrame[y_s:y_s+bS, x_s:x_s+bS]
    sim_temp = calcBlockSimilarity(slidingBlock, actualBlock)
    simx,simy = (x_s,y_s)
    for y in range(y_s, y_e-bS-1):
        for x in range(x_s, x_e-bS-1):
            slidingBlock = refFrame[y:y+bS, x:x+bS]
#            print 'actual', np.shape(actualBlock)
            sim = calcBlockSimilarity(slidingBlock, actualBlock)
            if sim_temp > sim:
#            print sim
                sim_temp = sim
                simx,simy = (x, y)
            
    motionVector = (simy - actY, simx - actX)
    return motionVector

def calcBlockSimilarity(refBlock, actualBlock): # inout is two 16x16 block
    subtr = refBlock - actualBlock
    subtr = np.abs(subtr)
    similarity = np.sum(subtr)
    return similarity
    
def compensateMotion(refImg, motionVectors):
    bS = 16 # blockSize
    shape = np.shape(refImg)    
    width = shape[1]
    height = shape[0]
    modImg = np.zeros(shape)
    for y in range(0, height/bS):
        for x in range(0, width/bS):
            motionX = motionVectors[y, x, 1]
            motionY = motionVectors[y, x, 0]
            modImg[y*bS:y*bS+bS, x*bS:x*bS+bS] = refImg[y*bS+motionY:y*bS+motionY+bS, x*bS+motionX:x*bS+motionX+bS]
    return modImg

## End of functions

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

outputFolder = 'C:\\Users\\Tomi\\Desktop\\ERASMUS_ISEL\\FELEV_KOZBEN\\CSM_szuletes\\project3-video_danika\\MV'
Iname, Iextension = IframeFileName.split('.')
cv2.imwrite('MV\\%s.jpeg' %Iname, Iframe, (cv2.cv.CV_IMWRITE_JPEG_QUALITY, 75))   

# calculating P_MV frames, saving them to JPEGs
IframeJPEG = cv2.imread(os.path.join(outputFolder, '%s.jpeg' %Iname)) 
# ide nem az Iname-et kéne beolvasni?? az a jpeg, előtte ott a tömörítés

# reading back the JPEG I frame for this
motionVectorsList = []
for (dirname, dirnames, filenames) in os.walk(dirName):
    for filename in filenames:
        if filename != IframeFileName:
            img = cv2.imread(os.path.join(dirname, filename))
            size = os.path.getsize(os.path.join(dirname, filename))
            origFileSizeList.append(float(size))
            origFrameList.append(img)
            motionVectors, modImg = calcMotionVectors(IframeJPEG, img)
            motionVectorsList.append(motionVectors)
            MVframe = img - modImg
            MVframe += 128 # for the pixel values to be between 0...255
#            print Pframe
            name, extension = filename.split('.')
            cv2.imwrite('MV\\%s.jpeg' %name, MVframe, (cv2.cv.CV_IMWRITE_JPEG_QUALITY, 75))
            
# read back I and MV frames, calculate energy, entropy and restore original frames
jpegEnergyList = []
jpegEntropyList = []  
restoredFrameList = []
#restoredDir = 'D:\\DOCUMENTS\\SULI\\ISEL\\CSM\\workspace\\video\\MV_restored'
i = 0 # for indexing the motionVectorsList
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
            motionVectors = motionVectorsList[i]
            modImg = compensateMotion(IframeJPEG, motionVectors)
            i += 1
            restoredFrame = imgJPEG + modImg
            restoredFrame -= 128
            cv2.imwrite('MV_restored\\%s.jpeg' %filename, restoredFrame)
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