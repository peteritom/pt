# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:26:03 2015

@author: Tomi
"""

import os
import cv
import cv2
import csv
import numpy as np
import numpy.matlib
import facerec_tools as ft
import scipy.io as sio

# Computing the F matrix
facesDir = "./faces\\mpeg7"
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

# classification of faces

faceslist = []
with open('./csvs\\tr_classes.csv', 'rb') as f:
    for line in f.readlines():
        l,name = line.strip().split(';')
        faceslist.append(name)
faces = map(int,faceslist)

# Calculating the mean face of all faces and the mean face of each classes
meanFace = np.mean(F_mx, axis=1) # total mean face


j = len(faceslist)-1
l = 0
c = list()
while j>=0:
    l = l+1
    if faces[j-1] != faces[j]:
        c.append(l)
        l = 0
    j = j-1
c.reverse()

classNum = len(c)
m = imgNum-classNum # eigenvector number

l = 0
meanFace_class = np.zeros((N,classNum))
for j in range(0,classNum):
    meanFace_class[:,j] = np.mean(F_mx[:,l:(l+c[j])], axis=1) # mean face of classes
    l = l + c[j]

# Scatter matrix

A = np.subtract(F_mx.transpose(), meanFace).transpose()
R = np.dot(A.transpose(),A)

# Calculating the eigen vectors/values
S_t = np.dot(A,A.transpose())
eps, v = np.linalg.eig(R)

# Bulding the W_PCA matrix
eps_m = np.argsort(eps)[::-1] 
eps_m = eps_m[0:m] # indexes of the m largest eigenvalues
V = v.transpose()[eps_m].transpose()
W_PCA = np.dot(A,V)
k = np.size(W_PCA[0,:]) - 1
while k>=0:
    W_PCA[:,k] = W_PCA[:,k] / np.linalg.norm(W_PCA[:,k])
    k = k - 1

Sb = 0
for i in range(0,classNum):
    # I = np.reshape(np.subtract(meanFace_class[:,i].transpose(), meanFace).transpose(),(N,1))
    I = np.reshape((meanFace_class[:,i]-meanFace),(N,1))
    Sb += c[i]*np.dot(I,I.transpose())

l = 0
S_i = list()
S_i_temp = np.zeros((N,N))

for i in range(0,classNum):
    for j in range(l,l+c[i]):
        # a_temp = np.reshape((F_mx[:,j]-meanFace_class[:,i]),(N,1))
        a_temp = np.reshape(np.subtract(F_mx[:,j].transpose(),meanFace_class[:,i].transpose()).transpose(),(N,1))
        S_i_temp += np.dot(a_temp,a_temp.transpose())
    l = l + c[i]
    S_i.append(S_i_temp)
    S_i_temp = np.zeros((N,N))

Sw = sum(S_i)
# compute the Sb_hat Sw_hat matrix

Sb_hat = np.dot(W_PCA.transpose(),np.dot(Sb,W_PCA))
Sw_hat = np.dot(W_PCA.transpose(),np.dot(Sw,W_PCA))


# compute eigenvectors from inv(Sb_hat)*Sw_hat

m_hat = classNum - 1
R_hat = np.dot(np.linalg.inv(Sw_hat),Sb_hat)

(eps_hat, v_hat) = np.linalg.eig(R_hat)
eps_m_hat = np.argsort(eps_hat)[::-1]
v_hat_real = np.zeros(np.shape(v_hat))
for i in range(0,len(v_hat[0,:])):
    v_hat_real[:,i] = v_hat[:,i].real
    
 

W_fld = v_hat_real[:,eps_m_hat[0:m_hat]]
i = np.size(W_fld[0,:]) - 1
while i>=0:
    W_fld[:,i] = W_fld[:,i] / np.linalg.norm(W_fld[:,i])
    i = i-1

y_hat = np.dot(W_fld.transpose(),np.dot(W_PCA.transpose(),A))


testFacesDir = "./faces\\test"
filename = "colinpowell0005_4.jpg"
testFace = cv2.imread(os.path.join(testFacesDir, filename))
gray = cv2.cvtColor(testFace, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('TestFaceWindow',1)
cv2.imshow('TestFaceWindow', gray)

mostSim = ft.faceEngine (gray.astype('float64'), meanFace, W_fld, W_PCA, 5, y_hat)

for i in range(1,6):
    winName = "MostSimilar %d" % i
    print filenames[mostSim[i-1][1]]
    x_rec = F_mx[:,mostSim[i-1][1]]
    cv2.namedWindow(winName,1)
    cv2.imshow(winName,x_rec.reshape((a,b)).astype('uint8'))

cv2.waitKey(0)