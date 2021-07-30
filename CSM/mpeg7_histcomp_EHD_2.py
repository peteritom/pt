# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:52:44 2015

"""

import numpy as np
import cv2
import cv
import os
import mpeg7_tools as mp
t_edge = 5
l = 0
img_hist = []
dirname = "C:\\Users\\Tomi\\Desktop\\ERASMUS_ISEL\\FELEV_KOZBEN\\ArtVision\\project\\MPEG_7_descriptor\\edh_img"
for (dirname, dirnames, filenames) in os.walk(dirname):
    for filename in filenames:
        img = mp.imgResize (dirname, filename)
        hist = mp.localHistComp(img, t_edge)
        img_hist.append(hist)
        l = l+1
        print "Database: %d/%d pictures processed..." % (l,len(filenames))
np.save( 'edh_hist.npy', img_hist)

l = 0
img_hist = []
dirname = "C:\\Users\\Tomi\\Desktop\\ERASMUS_ISEL\\FELEV_KOZBEN\\ArtVision\\project\\MPEG_7_descriptor\\edh_imgQ"
for (dirname, dirnames, filenames) in os.walk(dirname):
    for filename in filenames:
        img = mp.imgResize (dirname, filename)
        hist = mp.localHistComp(img, t_edge)
        img_hist.append(hist)
        l = l+1
        print "Query: %d/%d pictures processed..." % (l,len(filenames))
np.save( 'edh_histQ.npy', img_hist)

cv2.imshow("win2", img.astype('uint8'));

cv2.waitKey(0)