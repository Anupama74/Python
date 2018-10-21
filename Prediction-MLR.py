# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 20:59:39 2018

@author: Anu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from pandas.plotting import scatter_matrix
import seaborn as sns
df= pd.read_csv('utilities.csv')
#print(df.head())
df.describe()
##--------------------------------------------

data=df
data.plot(kind='box', subplots=True, sharex=False, sharey=False, figsize=(10,4), colormap='Spectral')  #Univariante
data.hist()        #Gaussian distribution
scatter_matrix(data)  #interaction b/n variables
plt.show()

data.plot(kind='density',subplots=True,sharex=False)
plt.show()

correlation = data.corr()
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlation, vmin=-1,vmax=1)
fig.colorbar(cax)
plt.show()
#data = data.corr()['Year'].sort_values()
#print(data)  ########

##-----------------------------------------------

x=df[['Domestic','Agriculture','Commercial','Industry','Others','Traction']]
y=df['Year']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=0)

regressor = LinearRegression()  
regressor.fit(x_train, y_train)
coeff_df = pd.DataFrame(regressor.coef_, x.columns, columns=['Coefficient'])  
print(coeff_df)
y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
#print(df)
df.plot(kind='density',sharex=True)
plt.xlabel('Years')
plt.title('Predicted graph')
plt.show()
#
sns.residplot(y_test,x_test['Domestic'],color='black',label='Actual data')
sns.residplot(y_pred,x_test['Domestic'],label='fitted data')
#plt.scatter(y_test,x_test['Commercial'],color='black',label='Actual data')
#plt.scatter(y_pred,x_test['Commercial'],color='red',label='fitted data')
plt.xlabel('Years')
plt.ylabel('Data in %')
plt.title('Evaluation data')
plt.legend()
plt.show()

##--------------------------------------
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('R2:', metrics.r2_score(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

