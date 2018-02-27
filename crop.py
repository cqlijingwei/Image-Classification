#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 01:54:14 2018

Crop out uncropped images.

@author: liuxiaoqin
"""

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import PIL
import shutil

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255

for filename in os.listdir("uncropped/"):
    shutil.copy2("uncropped/"+filename, "cropped/")

f = open('faceANDsize.txt','r')
for line in f.readlines():
    fn = line.split()[0]
    if os.path.isfile("cropped/"+fn):
        x1,y1,x2,y2=line.split()[1].split(",")
        try:
           im = imread("cropped/"+fn)
           face = im[int(y1):int(y2), int(x1):int(x2)]
           gray = rgb2gray(face)
           print fn
           img = imresize(gray, (32,32))
           imsave("cropped/"+fn,img)
        except (IndexError, IOError, ValueError) as e:
           os.remove("cropped/"+fn)
f.close()
