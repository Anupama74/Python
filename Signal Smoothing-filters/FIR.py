# -*- coding: utf-8 -*-
"""
Created on Sat May  27 15:41:01 2017

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
plt.plot (time,x,time,y,'g-')
plt.title('original signals for FIR filter')
plt.xlabel('Time')
plt.ylabel('X & Y')
plt.grid(True)

sample_rate = 500
nsamples = len(x)
nyq_rate = sample_rate/2
width = 5/nyq_rate
ripple_db= 60
N,beta = sp.signal.kaiserord (ripple_db,width)
cutoff = 30
taps = sp.signal.firwin (N,cutoff/nyq_rate,window=('kaiser',beta))
x1 = sp.signal.lfilter(taps,1,x)
y1 = sp.signal.lfilter(taps,1,y)

delay = 0.5*(N-1)/sample_rate

plt.figure(2)
plt.plot(time,x1,'.',time,y1,'b-')
plt.plot(time[N-1:]-delay,x1[N-1:],'r')
plt.title('Low pass FIR signals')
plt.xlabel('Time')
plt.ylabel('X & Y')
plt.grid(True)

plt.figure(3)
w,h = sp.signal.freqz (taps,worN = 30)
plt.plot(w*nyq_rate,h,'*-')
plt.xlabel('Frequency(hz)')
plt.ylabel('Gain')
plt.title('Frequency response')
plt.grid(True)
plt.show()