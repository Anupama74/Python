# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 16:08:43 2017

@author: Anu
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

sFile = 'AirQualityUCI1.csv' 
Data = pd.read_csv(sFile, sep =';')

C6H6 = Data['C6H6(GT)']
CO= Data['CO(GT)']
Date= Data['Date']    

N=len(Date)
Year = []; 
Month = [];
Day = [];
for i in range(N):
    day, month, year = Date[i].split('/')
    Day.append(day)
    Month.append(np.int32(month))
    Year.append(np.int32(year))

Years = np.unique(Year)
Month = np.array(Month)


i = 0;
MeanC6H6 = []
for year in Years:
    mask = np.int32(Year) == np.int32(year)
    C6H6_of_year = C6H6[mask]
    Month_of_year = Month[mask]
    Months = np.unique(Month_of_year)
    for month in Months:
        mask = (Month_of_year) == (month)
        print('years %s MOnths %s' %(year,month) )
        C6H6_of_month = C6H6_of_year[mask]
        mask = C6H6_of_month >0
        meanC6H6 = np.mean(C6H6_of_month[mask])
        print('Monthly Average %f' %(meanC6H6))
        MeanC6H6.append(meanC6H6)

plt.plot(MeanC6H6, '*')  
#-----------------------------------------------2 point
mask1 = C6H6 >0
mask2 = CO>0
mask = mask1 & mask2

C6H6 = C6H6[mask]
CO = CO[mask]

plt.figure(2)

plt.plot (C6H6, CO, '*')
p = np.polyfit(C6H6, CO, 1)
print ('p=', p)
CO_fitted_1 = np.polyval(p, C6H6)
plt.plot(C6H6,CO_fitted_1, 'r' )
var_resid = np.sum(np.power(CO-CO_fitted_1, 2))/(len(CO)-2)
## -------------------------3rd point
Year = Year[:7344]
Month = Month[:7344]

C6H6byMonth = {}
k=0
for year in Years:
    mask = np.int32(Year) == np.int32(year)
    C6H6_of_year = C6H6[mask]
    Month_of_year = Month[mask]
    Months = np.unique(Month_of_year)
    for month in Months:
        mask = (Month_of_year) == (month)
        print('years %s MOnth %s' %(year,month) )
        C6H6_of_month = C6H6_of_year[mask]
        mask = C6H6_of_month > 0
        # k = k + 1
        C6H6byMonth[k] = C6H6_of_month[mask]
        k = k + 1
        
[F, p] = stats.f_oneway(C6H6byMonth[0],C6H6byMonth[1],  C6H6byMonth[2],
 C6H6byMonth[3], C6H6byMonth[4],  C6H6byMonth[5],
C6H6byMonth[6], C6H6byMonth[7],  C6H6byMonth[8])

#--------------------------------------4th point
C6H6 = C6H6[:7109]
Month = Month[:7109]
Day = Day[:7109]
data = np.zeros((7109,3), dtype = np.float64)
data[:,0] = C6H6
data[:,1] = Month
data[:,2] = Day
df = pd.DataFrame(data, columns=['C6H6', 'Month', 'Day'])
formula = 'C6H6 ~ C(Month) + C(Day) + C(Month):C(Day)'
lm = ols(formula, df).fit()
print(anova_lm(lm))
#--------------------------------5th point
stats.ttest_ind(C6H6byMonth[6], C6H6byMonth[5])



