# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:55:07 2015

"""

import cv2
import numpy as np
import os

# functions for calculating entropy, energy, SNR
def computeSNR (originalImg, decodedImg):
    origFloat = np.float32(originalImg)
    decodedFloat = np.float32(decodedImg)
    energyOriginal = np.sum(np.power(1.*origFloat , 2)/np.size(origFloat))
    energyNoise = np.sum(np.power(1.*(origFloat - decodedFloat), 2)/np.size(origFloat) )
    return 10*np.log10(energyOriginal / energyNoise)

def computeEntropy(img):
    hist = cv2.calcHist([img],[0],None,[128],[0,128])
    # remove 0-elements           
    idxs = np.any(hist != 0.0, axis = 1)
    hist = hist[idxs]
#    print np.sum(hist) # should be 84480.0

    pi = hist/np.sum(hist)
    entropy = np.sum(pi*np.log2(1/pi))
    return entropy
    
def computeEnergy(img):
    energy = np.sum(np.power(img, 2)) / np.size(img)
    return energy