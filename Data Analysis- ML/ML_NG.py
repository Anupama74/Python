# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 21:41:07 2018

@author: Anu
"""

import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler, Normalizer, Binarizer, scale
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import metrics
df= pd.read_csv('utilities.csv')
df.head()
#Total=[row.Domestic+row.Commercial+row.Industry+row.Traction+row.Agriculture+row.Others for index, row in df.iterrows()]
#df['Total']=Total
#print(df.head())
#print(df.describe())


"""    ########
df.plot(kind='box', subplots=True, sharex=False, sharey=False)  #Univariante
df.hist()        #Gaussian distribution
scatter_matrix(df)  #interaction b/n variables
plt.show()

df.plot(kind='density',subplots=True,sharex=False)
plt.show()

correlation = df.corr()
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlation, vmin=-1,vmax=1)
fig.colorbar(cax)
#data = df.corr()['Year'].sort_values()
#print(data)  ########
"""

"""
scaled = StandardScaler().fit(df)
df= scaled.transform(df)         #Standardizing
scaler=Normalizer().fit(df)
df= scaler.transform(df)       #Normalizing
biny = Binarizer(threshold=0.0).fit(df)
df=biny.transform(df)         #Binarizing"""
##----------------------------------------------------------------

x=df[['Domestic','Agriculture','Commercial','Industry','Others','Traction']]
y=df['Year']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=0)

rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
y_rbf = rbf.fit(x_train, y_train).predict(x_test)
#print(y_rbf)
ndf = pd.DataFrame({'Actual': y_test, 'Predicted': y_rbf})
#print(ndf)

ndf.plot(kind='density',sharex=True)
plt.xlabel('Years')
plt.title('Predicted graph')
plt.show()


sns.residplot(y_test,x_test['Domestic'],color='black',label='Actual data')
sns.residplot(y_rbf,x_test['Domestic'],label='fitted data')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_rbf))  
print('R2:', metrics.r2_score(y_test,y_rbf))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_rbf))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_rbf))) 
