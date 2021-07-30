# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:56:23 2015

"""

import numpy as np
import cv2
import cv
import os
import mpeg7_tools as mp


l = 0
imgDir = "./project\\MPEG_7_descriptor\\pictures"
#imgDir = "./project\\MPEG_7_descriptor\\edh_img"
for (dirname, dirnames, filenames) in os.walk(imgDir):
    hist = np.zeros((256,len(filenames)))
    for filename in filenames:
        img = cv2.imread(os.path.join(dirname, filename))
        hist[:,l] = mp.colorDescr (img)
        l = l+1
        print "Database: %d/%d pictures processed..." % (l,len(filenames))
l = 0        
dirname = "./project\\MPEG_7_descriptor\\queryImages"
#dirname = "./project\\MPEG_7_descriptor\\edh_imgQ"
for (dirname, dirnames, filenames) in os.walk(dirname):
    histQ = np.zeros((256,len(filenames)))
    for filename in filenames:
        img = cv2.imread(os.path.join(dirname, filename))
        histQ[:,l] = mp.colorDescr (img)
        l = l+1
        print "Query: %d/%d pictures processed..." % (l,len(filenames))
"""
imgDir = "./project\\MPEG_7_descriptor\\pictures"
img = cv2.imread(os.path.join(imgDir, 'pyramid.jpg'))
file_obj = open('image_histograms.txt','r')
hist = file_obj.read()
print hist.find('\n')
"""
np.save( 'hist.npy', hist)
np.save( 'histQ.npy', histQ)
cv2.imshow("win1", img.astype('uint8'));
cv2.waitKey(0)