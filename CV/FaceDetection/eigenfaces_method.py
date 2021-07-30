# -*- coding: utf-8 -*-

"""
Created on Wed Mar 11 20:11:31 2015

"""

import os
import cv
import cv2
import csv
import numpy as np
import scipy.misc
import facerec_tools as ft


# Computing the F matrix
facesDir = "./mpeg7"
# Parameters
i = 0
a = 56
b = 46
N = a * b
for (dirname, dirnames, filenames) in os.walk(facesDir):
    imgNum = len(filenames)
    F_mx = np.zeros((N,imgNum))
    for filename in filenames:
        img = cv2.imread(os.path.join(dirname, filename))
        F_mx[:,i] = img[:,:,0].reshape((1,N))
        i = i+1


# Calculating the mean face of each classes, and the mean face of classes
meanFace = np.mean(F_mx, axis=1)
cv2.imshow('meanface',meanFace.reshape((a,b)).astype('uint8'))
# Calculating the A & S,R matrix
A_mx = np.subtract(F_mx.transpose(), meanFace).transpose()
S_mx = np.dot(A_mx,A_mx.transpose())
R_mx = np.dot(A_mx.transpose(),A_mx)

# Calculating the eigen vectors/values
eps, v = np.linalg.eig(R_mx)

MSE_wr = np.zeros((imgNum-1,1))
savefold = './results6_2'
l = 21
# video = cv2.VideoWriter('video.avi',-1,1,(a,b))
for m in range(1,imgNum):
# Bulding the W matrix
    eps_m = np.argsort(eps)[::-1] 
    eps_m = eps_m[0:m] # indexes of the m largest eigenvalues
    V_mx = v.transpose()[eps_m].transpose()
    W_mx = np.dot(A_mx,V_mx)
    k = np.size(W_mx[0,:]) - 1
    while k>=0:
        W_mx[:,k] = W_mx[:,k] / np.linalg.norm(W_mx[:,k])
        k = k - 1
# projection values, reconstructed faces and error computation
    Y_mx = np.dot(W_mx.transpose(),A_mx)
    ab = np.dot(W_mx,Y_mx)
    x_hat = np.add(ab.transpose(),meanFace).transpose() # reconstructed face
    """
    # classification test
        img_test = cv2.imread('./hugochavez0005_8.jpg')
        f_test = img_test[:,:,0].reshape((1,N))
        ac_test = (f_test - meanFace).astype('float64')
        y_test = np.dot(W_mx.transpose(),ac_test.transpose())
        par = 3
        test_class = ft.faceEngine (y_test, par, Y_mx)
    for i in range(0,len(test_class)):
        testface = F_mx[:,test_class[i][1]]
        testface_re = testface.reshape((a,b)).astype('uint8')
        WinName = "Class test{:.0f}".format(i)
        cv2.imshow(WinName,testface_re)
    """
    E_mx = np.zeros((N,imgNum)) # absolute error matrix
    for i in range(0,imgNum):
        E_mx[:,i] = np.abs(F_mx[:,i]-x_hat[:,i])
    MSE = np.zeros((imgNum,1))
    MSE = ((F_mx - x_hat) ** 2).mean(axis=1) # MSE error matrix
    
# test
# video = cv2.VideoWriter('video.avi',-1,1,(a,b))
    testPic = x_hat[:,l]
    testPic_re = testPic.reshape((a,b)).astype('uint8')
    cv2.imshow('Test',testPic_re)
    scipy.misc.imsave(savefold + '\\r%d.jpg' % m, testPic_re)
    
    
    errFace = E_mx[:,l]
    errFace_re = errFace.reshape((a,b)).astype('uint8')
# cv2.imshow('Error Face',errFace_re)
    MSE_wr[m-1] = MSE[l]
np.savetxt(savefold + '\\MSE_by_m.txt', MSE_wr)
cv2.waitKey(0)