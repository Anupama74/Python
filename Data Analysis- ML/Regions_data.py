# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 03:09:01 2018

@author: Anu
"""

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics

df= pd.read_csv('datafile.csv')
#print(df.head())
df.describe()
datax=df
correlation = datax.corr()
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlation, vmin=-1,vmax=1)
fig.colorbar(cax)
plt.show()
data=[('scale',StandardScaler()),('polynomial',PolynomialFeatures(degree=2)),('mode',LinearRegression())]
pipe=Pipeline(data)
x=df[['y-2007','y-2008','y-2009','y-2010','y-2011','y-2012']]
y=df['Region']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=0)
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
otp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#print(otp)
otp.plot(kind='density',sharex=True)
plt.xlabel('Regions')
plt.title('Predicted graph')
plt.show()
print('R2:', metrics.r2_score(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))