#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:42:09 2018

Multiclass classification

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
    x = np.vstack( (np.ones((1,x.shape[1])), x))
    return sum(sum( (np.dot(theta.T,x)-y) ** 2,axis=1))/2/x.shape[1]

def df(x, y, theta):
    x = np.vstack( (np.ones((1,x.shape[1])), x))
    return np.dot(x,(np.dot(theta.T, x)-y).T)/x.shape[1]

def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 100000
    iter  = 0
    while np.linalg.norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        iter += 1
    return t

delta=1e-8
N=5
K=6
M=7
for i in range(0,6):
    theta = np.random.rand(N,K)
    x=np.random.rand(N-1,M)
    y=np.random.rand(K,M)
    theta=np.random.rand(N,K)
    gradient = df(x,y,theta)
    fdAppro = (f(x,y,theta+delta)-f(x,y,theta))/delta
    diff = abs(sum(sum(gradient))-fdAppro)
    print ("Approximation ,gradient, abusolute difference:")
    print fdAppro,sum(sum(gradient)),diff,'\n'
    
act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

braccoTrain = []
gilpinTrain = []
harmonTrain = []
baldwinTrain = []
haderTrain = []
carellTrain = []
braccoValid = []
gilpinValid = []
harmonValid = []
baldwinValid = []
haderValid = []
carellValid = []

for filename in os.listdir("training/"):
    if 'bracco' in filename:
        braccoTrain.append(imread("training/"+filename)/255.0)
    if 'gilpin' in filename:
        gilpinTrain.append(imread("training/"+filename)/255.0)
    if 'harmon' in filename:
        harmonTrain.append(imread("training/"+filename)/255.0)
    if 'baldwin' in filename:
        baldwinTrain.append(imread("training/"+filename)/255.0)
    if 'hader' in filename:
        haderTrain.append(imread("training/"+filename)/255.0)
    if 'carell' in filename:
        carellTrain.append(imread("training/"+filename)/255.0)
        
for filename in os.listdir("validation/"):
    if 'bracco' in filename:
        braccoValid.append(imread("validation/"+filename)/255.0)
    if 'gilpin' in filename:
        gilpinValid.append(imread("validation/"+filename)/255.0)
    if 'harmon' in filename:
        harmonValid.append(imread("validation/"+filename)/255.0)
    if 'baldwin' in filename:
        baldwinValid.append(imread("validation/"+filename)/255.0)
    if 'hader' in filename:
        haderValid.append(imread("validation/"+filename)/255.0)
    if 'carell' in filename:
        carellValid.append(imread("validation/"+filename)/255.0)
        
theta = np.zeros((1025,6))
xT =np.zeros((1024,0))
yT =np.zeros((6,0))

for i in range(0,70):
    try:
        xT = np.hstack((xT,np.reshape(braccoTrain[i],(1024,1))))
        yT = np.hstack((yT,np.array([[ 1.],[ 0.],[ 0.],[ 0.],[ 0.],[ 0.]])))
    except IndexError:
        pass
    try:
        xT = np.hstack((xT,np.reshape(gilpinTrain[i],(1024,1))))
        yT = np.hstack((yT,np.array([[ 0.],[ 1.],[ 0.],[ 0.],[ 0.],[ 0.]])))
    except IndexError:
        pass
    try:
        xT = np.hstack((xT,np.reshape(harmonTrain[i],(1024,1))))
        yT = np.hstack((yT,np.array([[ 0.],[ 0.],[ 1.],[ 0.],[ 0.],[ 0.]])))
    except IndexError:
        pass
    try:
        xT = np.hstack((xT,np.reshape(baldwinTrain[i],(1024,1))))
        yT = np.hstack((yT,np.array([[ 0.],[ 0.],[ 0.],[ 1.],[ 0.],[ 0.]])))
    except IndexError:
        pass
    try:
        xT = np.hstack((xT,np.reshape(haderTrain[i],(1024,1))))
        yT = np.hstack((yT,np.array([[ 0.],[ 0.],[ 0.],[ 0.],[ 1.],[ 0.]])))
    except IndexError:
        pass
    try:
        xT = np.hstack((xT,np.reshape(carellTrain[i],(1024,1))))
        yT = np.hstack((yT,np.array([[ 0.],[ 0.],[ 0.],[ 0.],[ 0.],[ 1.]])))
    except IndexError:
        pass

theta = grad_descent(f,df,xT,yT,theta,1e-3)
imsave("multiclassification/theta1(bracco).jpg", np.reshape(theta[1:1025,0],(32,32)))
imsave("multiclassification/theta2(gilpin).jpg", np.reshape(theta[1:1025,1],(32,32)))
imsave("multiclassification/theta3(harmon).jpg", np.reshape(theta[1:1025,2],(32,32)))
imsave("multiclassification/theta4(baldwin).jpg", np.reshape(theta[1:1025,3],(32,32)))
imsave("multiclassification/theta5(hader).jpg", np.reshape(theta[1:1025,4],(32,32)))
imsave("multiclassification/theta6(carell).jpg", np.reshape(theta[1:1025,5],(32,32)))
        
"""Print the result"""
a1Tcount=0.00
a1T=0
a2Tcount=0.00
a2T=0
a3Tcount=0.00
a3T=0
a4Tcount=0.00
a4T=0
a5Tcount=0.00
a5T=0
a6Tcount=0.00
a6T=0
a1Vcount=0.00
a1V=0
a2Vcount=0.00
a2V=0
a3Vcount=0.00
a3V=0
a4Vcount=0.00
a4V=0
a5Vcount=0.00
a5V=0
a6Vcount=0.00
a6V=0
for bT in braccoTrain:
    bT=np.reshape(bT,(1024,1))
    a1T+=1
    if (np.argmax(h(bT,theta))==0):
        a1Tcount+=1
for bT in gilpinTrain:
    bT=np.reshape(bT,(1024,1))
    a2T+=1
    if (np.argmax(h(bT,theta))==1):
        a2Tcount+=1
for bT in harmonTrain:
    bT=np.reshape(bT,(1024,1))
    a3T+=1
    if (np.argmax(h(bT,theta))==2):
        a3Tcount+=1
for bT in baldwinTrain:
    bT=np.reshape(bT,(1024,1))
    a4T+=1
    if (np.argmax(h(bT,theta))==3):
        a4Tcount+=1
for bT in haderTrain:
    bT=np.reshape(bT,(1024,1))
    a5T+=1
    if (np.argmax(h(bT,theta))==4):
        a5Tcount+=1
for bT in carellTrain:
    bT=np.reshape(bT,(1024,1))
    a6T+=1
    if (np.argmax(h(bT,theta))==5):
        a6Tcount+=1
for bT in braccoValid:
    bT=np.reshape(bT,(1024,1))
    a1V+=1
    if (np.argmax(h(bT,theta))==0):
        a1Vcount+=1
for bT in gilpinValid:
    bT=np.reshape(bT,(1024,1))
    a2V+=1
    if (np.argmax(h(bT,theta))==1):
        a2Vcount+=1
for bT in harmonValid:
    bT=np.reshape(bT,(1024,1))
    a3V+=1
    if (np.argmax(h(bT,theta))==2):
        a3Vcount+=1
for bT in baldwinValid:
    bT=np.reshape(bT,(1024,1))
    a4V+=1
    if (np.argmax(h(bT,theta))==3):
        a4Vcount+=1
for bT in haderValid:
    bT=np.reshape(bT,(1024,1))
    a5V+=1
    if (np.argmax(h(bT,theta))==4):
        a5Vcount+=1
for bT in carellValid:
    bT=np.reshape(bT,(1024,1))
    a6V+=1
    if (np.argmax(h(bT,theta))==5):
        a6Vcount+=1
print ("Training set accuracy(bracco):"), "{:.1%}".format(a1Tcount/a1T),"\n"
print ("Training set accuracy(gilpin):"), "{:.1%}".format(a2Tcount/a2T),"\n"
print ("Training set accuracy(harmon):"), "{:.1%}".format(a3Tcount/a3T),"\n"
print ("Training set accuracy(baldwin):"), "{:.1%}".format(a4Tcount/a4T),"\n"
print ("Training set accuracy(hader):"), "{:.1%}".format(a5Tcount/a5T),"\n"
print ("Training set accuracy(carell):"), "{:.1%}".format(a6Tcount/a6T),"\n"
print ("Validation set accuracy(bracco):"), "{:.1%}".format(a1Vcount/a1V),"\n"
print ("Validation set accuracy(gilpin):"), "{:.1%}".format(a2Vcount/a2V),"\n"
print ("Validation set accuracy(harmon):"), "{:.1%}".format(a3Vcount/a3V),"\n"
print ("Validation set accuracy(baldwin):"), "{:.1%}".format(a4Vcount/a4V),"\n"
print ("Validation set accuracy(hader):"), "{:.1%}".format(a5Vcount/a5V),"\n"
print ("Validation set accuracy(carell):"), "{:.1%}".format(a6Vcount/a6V),"\n"
    