# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 22:59:26 2015

"""
import cv2
"""
video = cv2.VideoWriter('video.avi',-1,1,(46,56))
loadfold = './faces\\results1_1'
for i in range(0,27):
     img = cv2.imread(loadfold + '\\r%d.jpg' % i)
     video.write(img)
     print i
"""     
from PIL import Image, ImageSequence
import sys, os
loadfold = './faces\\results1_1'

im = Image.open(filename)
original_duration = im.info['duration']
frames = [frame.copy() for frame in ImageSequence.Iterator(im)]    
frames.reverse()

from images2gif import writeGif
    writeGif("reverse_" + os.path.basename(filename), frames, duration=original_duration/1000.0, dither=0)