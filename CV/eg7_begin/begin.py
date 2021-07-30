# -*- coding: utf-8 -*-
"""
Created on Tue Apr 07 21:57:15 2015

"""

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
dirname = "./faces"
filename = "rand.png"
img = cv2.imread(os.path.join(dirname, filename))
chans = cv2.split(img)
colors = ("b", "g", "r")
features = []
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    features.extend(hist)
 
    # plot the histogram
    plt.plot(hist, color = color)
    plt.xlim([0, 256])
plt.show()
print "flattened feature vector size: %d" % (np.array(features).flatten().shape)