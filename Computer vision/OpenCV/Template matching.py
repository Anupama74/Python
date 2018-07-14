# -*- coding: utf-8 -*-
"""
Created on Tue May 13 23:02:24 2017

@author: Anu
"""

#import sys, os, time
#import glob

import numpy as np
from matplotlib import pyplot as plt
plt.ion()

#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

import cv2
###################


img1=plt.imread('04.jpg')
img_fil = cv2.bilateralFilter(img1, 10, 75, 75)
img=cv2.cvtColor(img_fil, cv2.COLOR_BGR2GRAY)

template=np.uint8(plt.imread('template.png')[:,:,0]*255)

data=[] 
for size in np.arange(0.3, 1, 0.2):
    
    template1 = cv2.resize(template,None,fx=size, fy=size, interpolation = cv2.INTER_CUBIC)
    h, w = template1.shape

    res = cv2.matchTemplate(img,template1,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    data.append([h, w, max_val, max_loc[0],max_loc[1]])
    
    equal=cv2.flip(template1,1)
    res = cv2.matchTemplate(img,equal,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    data.append([h, w, max_val, max_loc[0],max_loc[1]])
    
data1= np.array(data)
column= np.argmax(data1[:,2])
h= data1[column, 0].astype(int)
w= data1[column, 1].astype(int)
max_val= data1[column, 2]
max_loc= data1[column, 3:5]
max_loc= max_loc.astype(int)
top_left= tuple(max_loc)

bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img1,top_left, bottom_right, 255,3)

plt.figure()
plt.imshow(img1)


