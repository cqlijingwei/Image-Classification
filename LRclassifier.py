#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:23:16 2018

Part 3

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

def h(x,theta):
    x = np.vstack( (np.ones((1, x.shape[1])), x))
    return np.dot(theta.T,x)
    
def f(x, y, theta):
    x = np.vstack( (np.ones((1, x.shape[1])), x))
    return sum( (y - np.dot(theta.T,x)) ** 2)

def df(x, y, theta):
    x = np.vstack( (np.ones((1, x.shape[1])), x))
    return -2*np.dot(x,(y-np.dot(theta.T, x)).T)

def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 5000000
    iter  = 0
    while np.linalg.norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        iter += 1
    return t

baldwinTrain = []
carellTrain = []
baldwinValid = []
carellValid = []
baldwinTest = []
carellTest = []

for filename in os.listdir("training/"):
    if 'baldwin' in filename:
        baldwinTrain.append(imread("training/"+filename)/255.0)
    if 'carell' in filename:
        carellTrain.append(imread("training/"+filename)/255.0)
        
"""training"""
theta = (np.random.rand(1025,1)-0.5)*2
BT = np.zeros((1024,0))
Y = np.hstack((np.ones((1,70)),-np.ones((1,70))))
for bT in baldwinTrain:
    BT = np.hstack((BT,np.reshape(bT,(1024,1))))
CT = np.zeros((1024,0))
for cT in carellTrain:
    CT = np.hstack((CT,np.reshape(cT,(1024,1))))
    
theta = grad_descent(f,df,np.hstack((BT,CT)),Y,theta,1e-5)
imsave("randImage/Full.jpg", np.reshape(theta[1:1025,0],(32,32)))

theta = np.zeros((1025,1))
Y = np.hstack((np.ones((1,70)),-np.ones((1,70))))
j=0
BT = np.zeros((1024,0))
for bT in baldwinTrain:
    j+=1
    BT = np.hstack((BT,np.reshape(bT,(1024,1))))
    if j == 2:
        BT2=BT

k=0
CT = np.zeros((1024,0))
for cT in carellTrain:
    k+=1
    CT = np.hstack((CT,np.reshape(cT,(1024,1))))
    if k == 2:
        CT2=CT

Y2 = np.hstack((np.ones((1,2)),-np.ones((1,2))))
theta2 = grad_descent(f,df,np.hstack((BT2,CT2)),Y2,theta,1e-5)      
imsave("thetaImage/Two.jpg", np.reshape(theta2[1:1025,0],(32,32)))
theta = grad_descent(f,df,np.hstack((BT,CT)),Y,theta,1e-5)
imsave("thetaImage/Full.jpg", np.reshape(theta[1:1025,0],(32,32)))

for filename in os.listdir("validation/"):
    if 'baldwin' in filename:
        baldwinValid.append(imread("validation/"+filename)/255.0)
    if 'carell' in filename:
        carellValid.append(imread("validation/"+filename)/255.0)
        
"""Print the result"""
bTcount=0.00
cTcount=0.00
bVcount=0.00
cVcount=0.00
for bT in baldwinTrain:
    bT=np.reshape(bT,(1024,1))
    print ("Training set cost(baldwin)"),f(bT,1,theta),"\\newline"
    if (h(bT,theta)>0):
        bTcount+=1
for cT in carellTrain:
    cT=np.reshape(cT,(1024,1))
    print ("Training set cost(carell)"),f(cT,-1,theta),"\\newline"
    if (h(cT,theta)<0):
        cTcount+=1
for bV in baldwinValid:
    bV=np.reshape(bV,(1024,1))
    print ("Validation set cost(baldwin)"),f(bV,1,theta),"\\newline"
    if (h(bV,theta)>0):
        bVcount+=1             
for cV in carellValid:
    cV=np.reshape(cV,(1024,1))
    print ("Validation set cost(carell)"),f(cV,-1,theta),"\\newline"
    if (h(cV,theta)<0):
        cVcount+=1
print ("Training set accuracy(baldwin):"), "{:.1%}".format(bTcount/len(baldwinTrain)),"\n"
print ("Training set accuracy(carell):"), "{:.1%}".format(cTcount/len(carellTrain)),"\n"
print ("Validation set accuracy(baldwin):"), "{:.1%}".format(bVcount/len(baldwinValid)),"\n"
print ("Validation set accuracy(carell):"), "{:.1%}".format(cVcount/len(carellValid)),"\n"
    
for filename in os.listdir("test/"):
    if 'baldwin' in filename:
        baldwinTest.append(imread("test/"+filename)/255.0)
    if 'carell' in filename:
        carellTest.append(imread("test/"+filename)/255.0)

