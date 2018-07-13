# -*- coding: utf-8 -*-
"""
Created on Sat May  13 18:51:08 2017

@author: Anu
"""

#import pandas as pd
import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
#from scipy import signal
from numpy import convolve

data = np.genfromtxt( 'ak.txt',comments = 'Time,X,Y',delimiter=",",skip_header=1, usecols = (0,1,2))

time = data[:,0]
x = data[:,1]
y = data[:,2]

def movingaverage (values , window):
    weights = np.repeat(1.0,window)/window
    sma = np.convolve(values,weights, 'valid')
    return sma

x1 = movingaverage (x,51)
y1 = movingaverage (y,51)

plt.figure(1)
plt.plot (time,x,time,y,'g-')
plt.title('original signals for moving average filter')
plt.xlabel('Time')
plt.ylabel('X & Y')
plt.show()

plt.figure(2)
plt.plot(time[len(time)-len(x1):],x1,time[len(time)-len(y1):],y1,'g-')
plt.title('filtered average moving signals')
plt.xlabel('Time')
plt.ylabel('X & Y')
plt.show()