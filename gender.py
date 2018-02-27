#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:22:58 2018

Classify the gender

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

act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

maleTrain = []
femaleTrain = []
maleValid = []
femaleValid = []
sizeM = []
sizeF = []

for filename in os.listdir("training/"):
    if ('baldwin' or 'hader' or 'carell') in filename:
        maleTrain.append(imread("training/"+filename)/255.0)
    if ('bracco' or 'gilpin' or 'harmon') in filename:
        femaleTrain.append(imread("training/"+filename)/255.0)

"""training"""
i=0
j=0
k=10
MT = np.zeros((1024,0))
for mT in maleTrain:
    mT = np.reshape(mT,(1024,1))
    MT = np.hstack((MT,np.reshape(mT,(1024,1))))
    i+=1
    if i>=k:
        sizeM.append(MT)
        k+=10
    
k=10
FT = np.zeros((1024,0))
for fT in femaleTrain:
    fT = np.reshape(fT,(1024,1))
    FT = np.hstack((FT,np.reshape(fT,(1024,1))))
    j+=1
    if j>=k:
        sizeF.append(FT)
        k+=10
    
for filename in os.listdir("validation/"):
    if ('baldwin' or 'hader' or 'carell') in filename:
        maleValid.append(imread("validation/"+filename)/255.0)
    if ('bracco' or 'gilpin' or 'harmon') in filename:
        femaleValid.append(imread("validation/"+filename)/255.0)

"""Print the result"""
mTcount=0.00
mVcount=0.00
i=0
iV=0
fTcount=0.00
fVcount=0.00
j=0
jV=0
k=10
fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure()
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)
ax3 = fig3.add_subplot(111)
ax4 = fig4.add_subplot(111)
for i in range(0, len(sizeM)):
    Y = np.hstack((np.ones((1,k)),-np.ones((1,k))))
    theta = grad_descent(f,df,np.hstack((sizeM[i],sizeF[i])),Y,np.zeros((1025,1)),1e-5)
    for mT in maleTrain:
        mT=np.reshape(mT,(1024,1))
        i+=1
        if (h(mT,theta)>0):
            mTcount+=1
    for fT in femaleTrain:
        fT=np.reshape(fT,(1024,1))
        j+=1
        if (h(fT,theta)<0):
            fTcount+=1
    for mV in maleValid:
        mV=np.reshape(mV,(1024,1))
        iV+=1
        if (h(mV,theta)>0):
            mVcount+=1
    for fV in femaleValid:
        fV=np.reshape(fV,(1024,1))
        jV+=1
        if (h(fV,theta)<0):
            fVcount+=1
    ax1.scatter(k,mTcount/i,c='black')
    ax2.scatter(k,fTcount/j,c='black')
    ax3.scatter(k,mVcount/iV,c='black')
    ax4.scatter(k,fVcount/jV,c='black')
    k+=10
    mTcount=0.00
    mVcount=0.00
    i=0
    iV=0
    fTcount=0.00
    fVcount=0.00
    j=0
    jV=0
ax1.set_ylabel("Accuracy Rate for Male(Training Set)")
ax1.set_xlabel("Training Size")
ax2.set_ylabel("Accuracy Rate for Female(Training Set)")
ax2.set_xlabel("Training Size") 
ax3.set_ylabel("Accuracy Rate for Male(Validation Set)")
ax3.set_xlabel("Training Size")
ax4.set_ylabel("Accuracy Rate for Female(Validation Set)")
ax4.set_xlabel("Training Size") 

"""Performance of 6 other actors"""
maleOther = []
femaleOther = []
maleA = ['vartan' ,'butler' , 'radcliffe']
femaleA = ['ferrera' , 'drescher' , 'chenoweth']

for filename in os.listdir("cropped/"):
    for a in maleA:
        if a in filename:
            try: 
                male = imread("cropped/"+filename)/255.0
                male=np.reshape(male,(1024,1))
                maleOther.append(male)
            except (IOError,ValueError) as e:
                pass
    for a in femaleA:
        if a in filename:
            try:
                female = imread("cropped/"+filename)/255.0
                female=np.reshape(female,(1024,1))
                femaleOther.append(female)
            except (IOError,ValueError) as e:
                pass

count=0.00
k=10
allSum=len(maleOther)+len(femaleOther)
for male in maleOther:
    if (h(male,theta)>0):
        count+=1

for female in femaleOther:
    female=np.reshape(female,(1024,1))
    if (h(female,theta)<0):
        count+=1
print ("6 other actors performance:"), "{:.1%}".format(count/allSum),"\n"