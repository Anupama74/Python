# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:33:06 2018

@author: Anu
"""

import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv('Sensordata.csv')
print(df.head())
df.describe()

df.plot(kind='density',subplots=True,sharex=False)
plt.show()

x=df['sensorID']
y1=df['Flow-rate']
y2=df['Temp']

ax1=plt.subplot(311)
plt.plot(x,y1,color='olive',markerfacecolor='blue')
plt.legend()
ax2=plt.subplot(312,sharex=ax1, sharey=ax1)
plt.plot(x,y2)
plt.legend()
plt.xlabel('Sensors based on ID')
"""
plt.plot( x, y1, marker='*', markerfacecolor='blue', markersize=12, color='purple', linewidth=4)
plt.plot( x, y2, marker='',color='skyblue', linewidth=2)
#plt.plot( 'x, 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
plt.legend()
"""
