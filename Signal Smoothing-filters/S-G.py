# -*- coding: utf-8 -*-
"""
Created on Sat May  13 14:25:05 2017

@author: Anu
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal

data = np.genfromtxt( 'ak.txt',comments = 'Time,X,Y',delimiter=",",skip_header=1, usecols = (0,1,2))

time = data[:,0]
x = data[:,1]
y = data[:,2]
plt.figure(1)
plt.plot (time,x,time,y,'g-')
plt.title('original signals for Savitzky-Golay filter')
plt.xlabel('Time')
plt.ylabel('X & Y')
plt.show()


    
x1= sp.signal.savgol_filter (x,51,3, deriv= 0)
y1= sp.signal.savgol_filter (y,51,3, deriv= 0)


plt.figure(2)
plt.plot(time,x1,time,y1,'r-')
plt.title('Smoothened Savitzky-Golay signals')
plt.xlabel('Time')
plt.ylabel('X & Y')
plt.show()