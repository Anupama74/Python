# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:31:03 2017

@author: Anu
"""

#import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal

data = np.genfromtxt( 'ak.txt',comments = 'Time,X,Y',delimiter=",",skip_header=1, usecols = (0,1,2))

time = data[:,0]
x = data[:,1]
y = data[:,2]
plt.figure(1)
plt.plot (time,x,time,y,'r-')
plt.title('original waves for median filter')
plt.xlabel('Time')
plt.ylabel('X & Y')
plt.show()

x1 = sp.signal.medfilt(x,51)
y1 = sp.signal.medfilt(y,51)

plt.figure(2)
plt.plot(time,x1,time,y1,'r-')
plt.title('filtered median waves')
plt.xlabel('Time')
plt.ylabel('X & Y')
plt.show()