# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:40:13 2015

"""
import cv
import cv2
import math
import numpy as np
import os

def subSample(img):
    (W,H,k) = np.shape(img)
    p = max(0,math.floor(math.log(math.sqrt(W*H),2)-7.5))
    K = np.uint(math.pow(2,p))
    a = W%K
    b = H%K
    a_ = np.ones((W,1))
    b_ = np.ones((H,1))
    W_ = len(a_[0:(W-a):K])
    H_ = len(b_[0:(H-b):K])
    newImg = np.zeros((W_,H_,k))
    for i in range (0,k):
        newImg[:,:,i] = img[0:(W-a):K,0:(H-b):K,i]
    return newImg
    
def hsv2hDS (H,S,V):
    Diff = S*V
    Sum = V*(2-S)/2
    h = H*255
    d = Diff*255
    s = Sum*255
    return h, d, s
    
def computeCodeDHS (Hue, Diff, Sum):
    Dind_0 = (Diff >= 0) & (Diff < 6)
    Dind_1 = (Diff >= 6) & (Diff < 20)
    Dind_2 = (Diff >= 20) & (Diff < 60)
    Dind_3 = (Diff >= 60) & (Diff < 110)
    Dind_4 = (Diff >= 110) & (255 >= Diff)
    Dind_34 = Dind_3 | Dind_4
    Dind_234 = Dind_2 | Dind_34
    D = np.zeros((len(Diff),1))
    H = np.zeros((len(Diff),1))
    S = np.zeros((len(Diff),1))
    D[Dind_0] = 0
    D[Dind_1] = 1
    D[Dind_2] = 2
    D[Dind_3] = 3
    D[Dind_4] = 4
    # ss0
    H[Dind_0] = 0
    sumRange = np.linspace(0,255,32)
    for i in range (0,31):
        sind = (Sum >= sumRange[i]) & (Sum <= sumRange[i+1])
        S[(sind & Dind_0)] = i
    # ss1
    hueRange = np.linspace(0,255,4)
    sumRange = np.linspace(0,255,8)
    for i in range (0,3):
        hind = (Hue >= hueRange[i]) & (Hue <= hueRange[i+1])
        H[(hind & Dind_1)] = i
    for i in range (0,7):
        sind = (Sum >= sumRange[i]) & (Sum <= sumRange[i+1])
        S[(sind & Dind_1)] = i
    # ss2,3,4
    hueRange = np.linspace(0,255,16)
    sumRange = np.linspace(0,255,8)
    for i in range (0,15):
        hind = (Hue >= hueRange[i]) & (Hue <= hueRange[i+1])
        H[(hind & Dind_234)] = i
    for i in range (0,7):
        sind = (Sum >= sumRange[i]) & (Sum <= sumRange[i+1])
        S[(sind & Dind_234)] = i
    NH = np.array([1,4,16,16,16])
    NS = np.array([32,8,4,4,4])
    Code = np.zeros((len(Diff),1))
    Code[Dind_0] = H[Dind_0]*NS[0] + S[Dind_0]
    codeSum = 0
    for i in range (0,1):
        codeSum = codeSum + NH[i]*NS[i]
    Code[Dind_1] = codeSum + H[Dind_1]*NS[1] + S[Dind_1]
    
    codeSum = 0
    for i in range (0,2):
        codeSum = codeSum + NH[i]*NS[i]
    Code[Dind_2] = codeSum + H[Dind_2]*NS[2] + S[Dind_2]

    codeSum = 0    
    for i in range (0,3):
        codeSum = codeSum + NH[i]*NS[i]
    Code[Dind_3] = codeSum + H[Dind_3]*NS[3] + S[Dind_3]

    codeSum = 0    
    for i in range (0,4):
        codeSum = codeSum + NH[i]*NS[i]
    Code[Dind_4] = codeSum + H[Dind_4]*NS[4] + S[Dind_4]
    return Code

def structuredHistogram (code):
    code = np.uint(code)
    stuctrHist = np.zeros((8,8))
    #stuctrHist = np.zeros((5,5))    
    hist = np.zeros((256,1))
    (a,b) = np.shape(code)
    for i in range(0,(a-7)):
        for j in range(0,(b-7)):
            stuctrHist = code[i:(8+i),j:(8+j)]
            #stuctrHist = image[(i*5):(4+5*i),(j*5):(4+5*j)]
            ind = np.unique(stuctrHist)
            hist[ind] = hist[ind]+1
    N = sum(hist)
    # Histogram non-uniform quantization + Coding
    hist = hist / N
    for k in range(0,256):
        if (hist[k] >= 0 and hist[k] < 0.000000001):
            #hist[k] = 0.000000001/2
            hist[k] = 0
        elif (hist[k] >= 0.000000001 and hist[k] < 0.037):
            histRange = np.linspace(0.000000001,0.037,25)
            codeRange = np.uint(np.linspace(1,25,25))
            for i in range (0,24):
                if (hist[k] >= histRange[i] and hist[k] < histRange[i+1]):
                    #hist[k] = (histRange[i] + histRange[i+1])/2
                    hist[k] = codeRange[i]
        elif (hist[k] >= 0.037 and hist[k] < 0.08):
            histRange = np.linspace(0.037,0.08,20)
            codeRange = np.uint(np.linspace(26,45,20))
            for i in range (0,19):
                if (hist[k] >= histRange[i] and hist[k] < histRange[i+1]):
                    #hist[k] = (histRange[i] + histRange[i+1])/2
                    hist[k] = codeRange[i]
        elif (hist[k] >= 0.08 and hist[k] < 0.195):
            histRange = np.linspace(0.08,0.195,35)
            codeRange = np.uint(np.linspace(46,80,35))
            for i in range (0,34):
                if (hist[k] >= histRange[i] and hist[k] < histRange[i+1]):
                    #hist[k] = (histRange[i] + histRange[i+1])/2
                    hist[k] = codeRange[i]
        elif (hist[k] >= 0.195 and hist[k] < 0.32):
            histRange = np.linspace(0.195,0.32,35)
            codeRange = np.uint(np.linspace(81,115,35))
            for i in range (0,34):
                if (hist[k] >= histRange[i] and hist[k] < histRange[i+1]):
                    #hist[k] = (histRange[i] + histRange[i+1])/2
                    hist[k] = codeRange[i]
        elif (hist[k] >= 0.32 and hist[k] <= 1.0):
            histRange = np.linspace(0.32,1.0,140)
            codeRange = np.uint(np.linspace(116,255,140))
            for i in range (0,139):
                if (hist[k] >= histRange[i] and hist[k] <= histRange[i+1]):
                    #hist[k] = (histRange[i] + histRange[i+1])/2
                    hist[k] = codeRange[i]
    return hist
       
def colorDescr (img):
    HsvImg = cv2.cvtColor(img, cv.CV_BGR2HSV)
    hsvImg = subSample(HsvImg)
    a = np.size(hsvImg[:,0,0])
    b = np.size(hsvImg[0,:,0])
    N = a*b
    Hue = hsvImg[:,:,0].reshape((1,N)).astype('float64')/255
    Sat = hsvImg[:,:,1].reshape((1,N)).astype('float64')/255
    Val = hsvImg[:,:,2].reshape((1,N)).astype('float64')/255
    HDiffSum = hsv2hDS (Hue,Sat,Val)
    H = HDiffSum[0]
    D = HDiffSum[1]
    S = HDiffSum[2]
    code = np.zeros((N,1))
    code = computeCodeDHS (H.transpose(), D.transpose(), S.transpose())
    code = np.reshape(code,(a,b))
    hist = np.reshape(structuredHistogram(code),(256,))
    return hist

def distanceMatrix(hist, histQ):
    N = np.size(hist,1) # number of database images
    M = np.size(histQ,1) # number of query images
    D = np.zeros((N,M))
    for i in range(0,M):
        for j in range(0,N):
            D[j,i] = np.sqrt(np.sum(np.power(hist[:,j]-histQ[:,i],2)))
    return D
            
def subImages (img, s):
    (W,H) = np.shape(img)
    a = W%s
    b = H%s
    W = W-a
    H = H-b
    W_ = np.floor(W/s)
    H_ = np.floor(H/s)
    Ws = []
    Hs = []
    for k in range (0,s+1):
        Ws.append(k*W_)
        Hs.append(k*H_)
    Ws = np.uint(Ws)
    Hs = np.uint(Hs)
    subImgs = []
    for i in range (0,s):
        for j in range (0,s):
            subImgs.append(img[Ws[i]:(Ws[i+1]),Hs[j]:(Hs[j+1])])
    return subImgs
    
def orient (imgBlock, t_edge):
    sBlock = subImages(imgBlock,2)
    a_k = sum(sBlock)/4.0
    fv = np.matrix('1.0 -1.0; 1.0 -1.0')
    fh = np.matrix('1.0 1.0; -1.0 -1.0')
    fd45 = np.matrix('1.41421356237 0; 0 -1.41421356237')
    fd135 = np.matrix('0 1.41421356237; -1.41421356237 0')
    fnd = np.matrix('2.0 -2.0; -2.0 2.0')
    mv = np.sum(np.abs(np.dot(a_k,fv))) # 0 class
    mh = np.sum(np.abs(np.dot(a_k,fh))) # 1 class
    md45 = np.sum(np.abs(np.dot(a_k,fd45))) # 2 class
    md135 = np.sum(np.abs(np.dot(a_k,fd135))) # 3 class
    mnd = np.sum(np.abs(np.dot(a_k,fnd))) # 4 class
    # uniform: 5 class
    M = np.array([mv,mh,md45,md135,mnd])
    ind = np.argsort(M)[::-1]
    if (M[ind[0]]>t_edge):
        res = ind[0]
    else:
        res = 5
    return res
    
def subImgHist (listOrient, sizesSubImg): # sizesSubImg should start with 0
    hist = [] # 
    for i in range (0,16):
        locHist = np.zeros((5,1))
        N = sizesSubImg[i+1]-sizesSubImg[i]
        for j in range(sizesSubImg[i],sizesSubImg[i+1]):
            if listOrient[j]!=5:
                locHist[listOrient[j]] = locHist[listOrient[j]] + 1                
        locHist = locHist/N        
        hist.append(locHist)
    return hist
    
def imgResize (dirname, filename):
    img = cv2.imread(os.path.join(dirname, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(544,544))
    return img

def localHistComp (img, t_edge):
    listSub = subImages(img,4)
    llImgBlocks = []
    for i in range(0,16):
        llImgBlocks.append(subImages(listSub[i],34))
    lOrient = []
    hList = []
    for i in range(0,16):
        for j in range(0,len(llImgBlocks[i])):
            res = orient(llImgBlocks[i][j],t_edge)
            lOrient.append(res) # tEdge -> fix érték-e??
    lengths = []
    for element in llImgBlocks:
        lengths.append( len(element))
    sizesSubImg = []
    sizesSubImg.append(0)
    for i in range (1,17):
        sizesSubImg.append(sum(lengths[0:i]))
    histo = subImgHist (lOrient, sizesSubImg)
    return histo, hList
    
def distEDHHist (hist, histQ):
    N = np.shape(hist) # number of database images
    M = np.shape(histQ) # number of query images
    N = N[0]
    M = M[0]
    D = np.zeros((N,M))
    for i in range(0,M):
        for j in range(0,N):
            D[j,i] = np.sum(np.abs(hist[j]-histQ[i]))
    return D