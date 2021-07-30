import os
import cv2
import numpy as np

def mouseHandler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        param[0] = True
        param[1] = np.array([x,y])

#facesDir = "C:\\Users\\Pedro\\geral\\work\\Isel\\Disciplinas\\va\\1112Sem2\\labs\\faces"
#facesDir = "C:\\Users\\Pedro\\geral\\work\\Isel\\Disciplinas\\va\\1314Sem2\\labs\\faces"
facesDir = "C:\\Users\\Tomi\\Desktop\\ERASMUS_ISEL\\FELEV_KOZBEN\\ArtVision\\project\\faces"

cv2.namedWindow('Face',1)
cv2.namedWindow('Gray',1)
num = 1
for (dirname, dirnames, filenames) in os.walk(facesDir):
    for filename in filenames:
        
        img = cv2.imread(os.path.join(dirname, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        param = [False, np.zeros((1,2))]
        eyes = []
        numOfPoints = 0;
        while numOfPoints < 2:
            cv2.setMouseCallback('Face', mouseHandler, param)
            cv2.imshow('Face',img)
            if param[0]:
                point = param[1]
                eyes.append(param[1])
                numOfPoints = numOfPoints + 1;
                print point
                cv2.circle(img, (int(point[0]), int(point[1])), 2, (0,255,0))
                param[0] = False
            key = cv2.waitKey(1)

        eye1pos = eyes[0]
        eye2pos = eyes[1]

        difpos = eye2pos-eye1pos
        distpos = np.sqrt(np.sum(np.power(difpos,2)))
        ang = np.arctan2(difpos[1], difpos[0])*180.0/np.pi
        scaleFact = 16.0/distpos
        imgSize = gray.shape
        rotMat = cv2.getRotationMatrix2D((imgSize[0]/2, imgSize[1]/2), ang, 1)
        grayT = cv2.warpAffine(gray,rotMat,(imgSize[1], imgSize[0]))
        eye1posA = np.append(eye1pos, 1)
        eye2posA = np.append(eye2pos, 1)
        eye1posAT = np.dot(rotMat, eye1posA)
        eye2posAT = np.dot(rotMat, eye2posA)
        bE = ((eye1posAT+eye2posAT)/2.0)*scaleFact
        grayTS = cv2.resize(grayT,(int(round(imgSize[1]*scaleFact)), int(round(imgSize[0]*scaleFact))))
        face = grayTS[bE[1]-23:bE[1]+33, bE[0]-23:bE[0]+23]
        cv2.imwrite("C:\\Users\\Tomi\\Desktop\\ERASMUS_ISEL\\FELEV_KOZBEN\\ArtVision\\project\\faces\\mpeg7\\other_mpeg%d.png" % num, face );
        num = num + 1
        cv2.imshow('Face',img)
        cv2.imshow('Gray', face)
        cv2.waitKey(0)        

cv2.destroyWindow('Face')
cv2.destroyWindow('Gray')
