# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:03:15 2015

"""
import numpy as np
    
def faceEngine (faceq, meanFace, W_fld, W_PCA, N, y_hat):
    faceq_reshaped = np.reshape(faceq, (np.size(faceq), 1))
    ac_faceq = faceq_reshaped - np.reshape(meanFace, (2576,1))
    y_faceq = np.dot(W_fld.transpose(), np.dot(W_PCA.transpose(), ac_faceq))
    dist = []
    for i in range (0,y_hat.shape[1]):
        eucl = np.sqrt(np.sum(np.power(y_hat[:,i]-y_faceq.transpose(),2)))
        dist.append((eucl,i)) 
    dist_sorted = sorted(dist, key=lambda tup: tup[0])
    return dist_sorted[0:N]