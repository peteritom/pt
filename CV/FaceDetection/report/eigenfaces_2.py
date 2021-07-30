# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:52:07 2015

"""

import os
import cv2
import numpy as np

facesDir = "./faces\\faceDataBaseN\\treino"
numOfFaces = 28
n = 2576 # 56*46 a MPEG7 szabvány szerint

F = np.zeros((n, numOfFaces))

# calculating F matrix
i = 0
for (dirname, dirnames, filenames) in os.walk(facesDir):
    for filename in filenames:
        img = cv2.imread(os.path.join(dirname, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgReshaped = np.reshape(gray, (np.size(gray), 1))  # images are reshaped
        F[:,i] = imgReshaped[:,0]
        i = i+1
        
# calculating the mean face    
mu = 0
i = numOfFaces - 1
while i>=0:
    mu = mu + F[:,i]
    i = i-1
mu = mu / numOfFaces

# displaying the mean face
meanface = np.reshape(mu, (56,46)).astype('uint8') # conversion from float to int because of cv2.imshow()
cv2.namedWindow('MeanFaceWindow', 1)
cv2.imshow('MeanFaceWindow', meanface)
        
# calculating A matrix (A = F - mu)
A = np.subtract(F.transpose(), mu).transpose() # transponation is needed, because substract() substracts rows
# calc R matrix
R = np.dot(A.transpose(), A)    # should be size of NxN, where N is the number of faces

# compute eigenvectors, eigenvalues of R
(eigVals, eigVectors) = np.linalg.eig(R)
sortedIndexes = np.argsort(eigVals)
maxIndex = np.argmax(eigVals)

m = numOfFaces - 1  # m is changeable
# putting the m eigenvectors belonging to the highest eigenvalues to matrix V
V = np.zeros((numOfFaces, m))
i = 1
while i<=m:
    V[:,i-1] = eigVectors[:,sortedIndexes[-i]] # sortedIndexes first element is the index for the smallest eigenValue, that's why the negative indexing
    i = i + 1
# compute the W matrix
W = np.dot(A,V)
# normalisation
i = np.size(W[0,:]) - 1
while i>=0:
    W[:,i] = W[:,i] / np.linalg.norm(W[:,i])
    i = i - 1

## checking if the W matrix is correct or not
#    testImage = np.reshape(W[:,0], (56,46)).astype('uint8')
#    cv2.namedWindow('TestImageWindow',1)
#    cv2.imshow('TestImageWindow', testImage)
#    # tesztelem, h egy-egy oszlop ortonormális-e
#    testDotProduct = np.dot(W[:,0],W[:,1])

# calculate the projection matrix Y
Y = np.dot(W.transpose(), A)

# reconstructing the faces
Xhat = np.zeros((n,numOfFaces))
Xhat = np.add(mu, np.dot(W, Y).transpose()).transpose()


# POINT 4)
# testing the reconstructed faces
testImage = np.reshape(Xhat[:,8], (56,46)).astype('uint8')
cv2.imshow('TestImageWindow', testImage)

# computing error between the original and the reconstructed face
error = np.zeros((n, numOfFaces))
i = 0
while i < numOfFaces:
#    error[:,i] = np.sqrt(np.power(F[:,i], 2) - np.power(Xhat[:,i], 2))
    error[:,i] = np.abs(F[:,i] - Xhat[:,i]) # if m is set to be lower then NumOfFaces-1, then the error increases, and the error face starts to become not only a black window
    i += 1
# error arc megjelenites teszt
testImage = np.reshape(error[:,8], (56,46)).astype('uint8')
cv2.namedWindow('ErrorFaceWindow', 1)
cv2.imshow('ErrorFaceWindow', testImage)
# checking if the error face is orthogonal to the face subspace
ortoCheck = np.zeros(numOfFaces)
for i in range(0, m):
    ortoCheck[i] = np.dot(W[:,i], error[:,i]) # if every value is 0, it's OK

# POINT 5)
# read in the class info from csv file
facesList = []
with open('./faces\\faceDataBaseN\\treino_classes.csv', 'rb') as f:
  #  reader = csv.reader(f)
    for line in f.readlines():
        l, name = line.strip().split(';')
        facesList.append((l,name))
    
faceClassList = [] # tuple list (classIndex, faceVector)
j = 0
for fileName, faceClass in facesList:
    faceClassList.append((faceClass, F[:,j]))
    j += 1
    
def faceEngine(faceq, N): # returns a list of N indices with the mos similar faces to faceq of the database
    # making the projection vector of the input face [subtract the mean face from the input face, then multiply by W]
    faceq_reshaped = np.reshape(faceq, (np.size(faceq), 1))    # this should be a (2576,1) vector
    ac_faceq = faceq_reshaped - np.reshape(mu, (2576,1))
#    ac_faceq = faceq - np.reshape(mu, (2576,1)) # this needed, if the input parameter is already reshaped
    y_faceq = np.dot(W.transpose(), ac_faceq)
    # calculate its Euclidean distance with the other y vectors
    eucDistances = [] # tupleList /w (eucDistance, index)
    for i in range(0, numOfFaces):
        euc = np.linalg.norm(y_faceq-Y[:,i])
        eucDistances.append((euc, i))
    # sort the tupleList
    sortedEucDistances = sorted(eucDistances, reverse=False) # from the least to the biggest
    # return the N indexes in a list
    indexList = []
    for i in range(0, N):
        print sortedEucDistances[i][1]
        indexList.append(sortedEucDistances[i][1])
    return indexList
    
# read the test face
testFacesDir = "./faces\\faceDataBaseN\\teste"
filename = "donaldrumsfeld0012_5.jpg"
testFace = cv2.imread(os.path.join(testFacesDir, filename))
gray = cv2.cvtColor(testFace, cv2.COLOR_BGR2GRAY)

testList = faceEngine(gray, 3)
        
testImage = np.reshape(F[:,testList[0]], (56,46)).astype('uint8')
cv2.namedWindow('TestImageWindow',1)
cv2.imshow('TestImageWindow', testImage)
    



cv2.waitKey(0)
