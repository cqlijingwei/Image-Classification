#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 10:56:51 2018

Part2, separate files into training set, validation set and test set

@author: liuxiaoqin
"""

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import os
from scipy.ndimage import filters
import urllib
import re
import shutil

act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

for a in act:
    files=[]
    for filename in os.listdir("cropped/"):
        try:
            str1 = re.findall(r"[a-zA-Z]+",filename.split(".")[0])[0]
            if str1 in a.lower():
                files.append(filename)
        except IndexError:
            pass
    j = len(files)-1
    for i in range(0,10):
        fn = files.pop(random.randint(0,j))
        shutil.copy2("cropped/"+fn, "validation/")
        j-=1
    for i in range(0,10):
        fn = files.pop(random.randint(0,j))
        shutil.copy2("cropped/"+fn, "test/")
        j-=1
    for i in range(0,70):
        try:
            fn = files.pop(random.randint(0,j))
            shutil.copy2("cropped/"+fn, "training/")
            j-=1
        except ValueError:
            continue
            