# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:36:05 2015

@author: Daniel
"""

# TODO: megszámolni, hogy egy-egy classban hány arc van, mert ez változó lehet
# balancednak jó, ha van, de 1-1 darabban eltérhet, az inter scatter matrix képletében ez a kis ni
# szal revizionálni kellene az adattárolást a classoknál, class meanFace számításnál

# TODO: faceNumOfClassesDict nem mukodott helyesen, kis hack j = 1-gyel, gondold át 
import os
import cv2
import numpy as np

facesDir = "./faces\\faceDataBaseN\\treino"
numOfFaces = 28
n = 2576

# calculating F matrix
F = np.zeros((n, numOfFaces))
i = 0
for (dirname, dirnames, filenames) in os.walk(facesDir):
    for filename in filenames:
        img = cv2.imread(os.path.join(dirname, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgReshaped = np.reshape(gray, (np.size(gray), 1))  # images are reshaped
        F[:,i] = imgReshaped[:,0]
        i = i+1
numOfFaces = len(filenames) # N

# calculating the mean face    
mu = 0
i = numOfFaces - 1
while i>=0:
    mu += F[:,i]
    i = i-1
meanFace = mu / float(numOfFaces)
#meanFace = np.mean(F, axis = 1) # another method

# displaying the mean face
#meanFaceImg = np.reshape(meanFace, (56,46)).astype('uint8') # conversion from float to int because of cv2.imshow()
#cv2.namedWindow('MeanFaceWindow', 1)
#cv2.imshow('MeanFaceWindow', meanFaceImg)

#########################
# Classification
# classification of faces
#########################

# read in the class info from csv file
#faceClassList = [] # tuple list (classIndex, faceVector)
facesPerClassDict = {} # its elements are like: (classIndex, numOfFacesInClass)
with open('./faces\\faceDataBaseN\\treino_classes.csv', 'rb') as f:
  #  reader = csv.reader(f)
    i = 0
    facesPerClass = 0
    prevClassIndex = 1 # in the .CSV, the the classes have to start with index '1'
    for line in f.readlines():
        fileName, faceClass = line.strip().split(';')
        faceClass = int(faceClass)
#        faceClassList.append((faceClass, F[:,i]))
        # count the faces belonging to a class
        if faceClass != prevClassIndex:
            facesPerClassDict[prevClassIndex] = facesPerClass
            facesPerClass = 1
        else:
            facesPerClass += 1
        prevClassIndex = faceClass # renew the previous index stored
        # end of cycle
        i += 1
    facesPerClassDict[prevClassIndex] = facesPerClass # this is needed because of the last class (EoF reached, no new line is read)
    
# calculating the mean faces of the several classes    
numOfClasses = len(facesPerClassDict)
classMeanFaces = np.zeros((n, numOfClasses))
index = 0
for i in range(0, numOfClasses):
    classMeanFaces[:,i] = np.mean(F[:, index:(index+facesPerClassDict[i+1])], axis=1) # the faces of a class are after each other in F, so we count the mean of a part of F for each class
    index += facesPerClassDict[i+1] # move index according to the number of the current class' faces
    
#cv2.namedWindow('ClassMeanFaceWindow', 1)
#cv2.imshow('ClassMeanFaceWindow', np.reshape(classMeanFaces[:,0], (56,46)).astype('uint8'))

#####################
# calculate the total scatter matrix St   
A = np.subtract(F.transpose(), meanFace).transpose()
St = np.dot(A, A.transpose())
Rt = np.dot(A.transpose(), A)

# compute eigenvectors, eigenvalues of Rt
(eigVals, eigVectors) = np.linalg.eig(Rt)
m = numOfFaces - numOfClasses  # m is changeable
sortedIndexes = np.argsort(eigVals)
maxIndex = np.argmax(eigVals)
# putting the m eigenvectors belonging to the highest eigenvalues to matrix V
V = np.zeros((numOfFaces, m))
i = 1
while i<=m:
    V[:,i-1] = eigVectors[:,sortedIndexes[-i]] # sortedIndexes first element is the index for the smallest eigenValue, that's why the negative indexing
    i = i + 1
    
# compute the W matrix
W = np.dot(A, V)
# normalisation
i = np.size(W[0,:]) - 1
while i>=0:
    W[:,i] = W[:,i] / np.linalg.norm(W[:,i])
    i = i - 1

# calculate the inter scatter matrix Sb (NxN) - between class scatter
i = numOfClasses - 1
Sb = 0
while i>=0:
    # keplet a facerecog pdf 23. o. aljan
#    print faceNumOfClassesDict[i] 
    tempDiff = np.subtract(classMeanFaces[:,i].transpose(), meanFace).transpose() # mu_i - mu
    tempDiff = np.reshape(tempDiff, (n, 1))
    Sb += facesPerClassDict[i+1] * np.dot(tempDiff, tempDiff.transpose()) # classes are from 1...numOfClasses in the dictionary, that's why the +1
    i -= 1
   
## calculate the intra scatter matrix Sw (NxN) - within-class scatter
## Si means: 'covariance of class members and class meanFace'  
SiList = []
index = 0
for i in range(0, numOfClasses): # iterate through the classes
    Si = 0
    for j in range(index, index+facesPerClassDict[i+1]): # +1 in the dictionary index is needed, because the classes are numbered from 1..
        temp = np.reshape(F[:,j] - classMeanFaces[:,i], (n,1))
        Si += np.dot(temp, temp.transpose()) # classes in faceClassList are from 1..., but the meanFaceList indexes are from 0.., that's why the '-1' is needed
    SiList.append(Si)
    index += facesPerClassDict[i+1]
  
Sw = 0  
for Si in SiList:
    Sw += Si   

# calculate Sb_ and Sw_ matrices
Sb_ = np.dot(np.dot(W.transpose(), Sb), W)
#print Sb_
Sw_ = np.dot(np.dot(W.transpose(), Sw), W)
#print 'SW_ matrix: ' ,Sw_

# determining the c-1 larger eigenvectors
asd = np.dot(np.linalg.inv(Sw_), Sb_)
    
# compute eigenvectors, eigenvalues of asd
(eigVals, eigVectors) = np.linalg.eig(asd)
sortedIndexes = np.argsort(eigVals)[::-1] # argsort sorts from min to max, need to reverse it with [::-1]

eigVectorsReal = np.zeros(np.shape(eigVectors))
for i in range(0, len(eigVectors[0,:])):
    eigVectorsReal[:,i] = eigVectors[:,i].real
    
m_ = numOfClasses - 1  # m is changeable

# putting the m eigenvectors belonging to the highest eigenvalues to matrix V
Wfld = eigVectorsReal[:, sortedIndexes[0:m-1]]
# normalistaion
i = np.size(Wfld[0,:]) - 1
while i>=0:
    Wfld[:,i] = Wfld[:,i] / np.linalg.norm(Wfld[:,i])
    i = i - 1

Y = np.dot(Wfld.transpose(), np.dot(W.transpose(), A))

# CLASSIFICATION
def faceEngine(faceq, N): # returns a list of N indices with the most similar faces to faceq of the database
    # making the projection vector of the input face [subtract the mean face from the input face, then multiply by W]
    faceq_reshaped = np.reshape(faceq, (np.size(faceq), 1))    # this should be a (2576,1) vector
    print 'faceq_reshaped shape', faceq_reshaped.shape
    ac_faceq = faceq_reshaped - np.reshape(meanFace, (2576,1))
#    ac_faceq = faceq - np.reshape(mu, (2576,1)) # this needed, if the input parameter is already reshaped
    y_faceq = np.dot(Wfld.transpose(), np.dot(W.transpose(), ac_faceq))
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
    return indexList, eucDistances, sortedEucDistances
    
# read the test face
testFacesDir = "./faces\\faceDataBaseN\\teste"
filename = "donaldrumsfeld0012_5.jpg"
testFace = cv2.imread(os.path.join(testFacesDir, filename))
gray = cv2.cvtColor(testFace, cv2.COLOR_BGR2GRAY)
#testFaceReshaped = np.reshape(gray, (np.size(gray), 1))
cv2.namedWindow('TestFaceWindow',1)
cv2.imshow('TestFaceWindow', gray)

testList, eucDistances, sortedEucDistances = faceEngine(gray, 5)
        
classifiedImage = np.reshape(F[:,testList[2]], (56,46)).astype('uint8')
#testImage = np.reshape(F[:,7], (56,46)).astype('uint8')
cv2.namedWindow('ClassificationImageWindow',1)
cv2.imshow('ClassificationImageWindow', classifiedImage)









print 'kesz'      
cv2.waitKey(0)