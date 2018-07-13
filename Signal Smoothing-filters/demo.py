# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 18:49:49 2017

@author: Anu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
data = np.genfromtxt('stockholm_td_adj.dat')

fig, ax = plt.subplots(figsize=(14,4))
ax.plot(data[:,0]+data[:,1]/12.0+data[:,2]/365, data[:,5])
ax.axis('tight')
ax.set_title('tempeatures in Stockholm')
ax.set_xlabel('year')
ax.set_ylabel('temperature (C)');



X = data[:,5]


x = np.linspace(-3, 3, 100)
norm_pdf = lambda x: (1/np.sqrt(2 * np.pi)) * np.exp(-x * x / 2)
y = norm_pdf(x)

fig, ax = plt.subplots(1, 1, sharex=True)
ax.plot(x, y)
ax.fill_between(x, 0, y, where = x > 1.96)
ax.fill_between(x, 0, y, where = x < -1.96)
plt.title('Rejection regions for a two-tailed hypothesis test at 95% confidence')
plt.xlabel('x')
plt.ylabel('p(x)');



test_statistic = (X.mean() - 0)/(X.std())
print('t test statistic: ', test_statistic)


from scipy.stats import t


p_val = 2 * (1 - t.cdf(test_statistic, len(X) - 1))
print('P-value is: ', p_val)



X2001 = data[data[:,0] == 2000,5]
X2002 = data[data[:,0] == 2001,5]

plt.hist(X2001)
plt.hist(X2002)


mu_2001 = X2001.mean()
mu_2002 = X2002.mean()
s_2001 = X2001.std()
s_2002 = X2002.std()
n_2001 = len(X2001)
n_2002 = len(X2002)

test_statistic = ((mu_2001 - mu_2002) - 0)/((s_2001**2/n_2001) + (s_2002**2/n_2002))**0.5
df = ((s_2001**2/n_2001) + (s_2002**2/n_2002))**2/(((s_2001**2 / n_2001)**2 /n_2001)+((s_2001**2 / n_2001)**2/n_2001))

print('t test statistic: ', test_statistic)
print('Degrees of freedom (modified): ', df)

#t_stat, p_value = scipy.stats.ttest_1samp(X, 0)
#print("paprastas 2 imciu")
t_stat, p_value = scipy.stats.ttest_ind(X2001, X2002)
print("porinis")
print(t_stat)
print(p_value)
scipy.stats.ttest_ind
scipy.stats.ttest_rel
scipy.stats.t.ppf(0.975, len(x)-1)


#from scipy.stats import chi2
#
#
#x = np.linspace(0, 8, 100)
#y_1 = chi2.pdf(x, 1)
#y_2 = chi2.pdf(x, 2)
#y_3 = chi2.pdf(x, 3)
#y_4 = chi2.pdf(x, 4)
#y_6 = chi2.pdf(x, 6)
#y_9 = chi2.pdf(x, 9)
#
#fig, ax = plt.subplots()
#ax.plot(x, y_1, label = 'k = 1')
#ax.plot(x, y_2, label = 'k = 2')
#ax.plot(x, y_3, label = 'k = 3')
#ax.plot(x, y_4, label = 'k = 4')
#ax.plot(x, y_6, label = 'k = 6')
#ax.plot(x, y_9, label = 'k = 9')
#ax.legend()
#plt.title('Chi-Square distribution with k degrees of freedom')
#plt.xlabel('x')
#plt.ylabel('p(x)');
#
#
#start = "2015-01-01"
#end = "2016-01-01"
#pricing_sample = get_pricing("MSFT", start_date = start, end_date = end, fields = 'price')
#returns_sample = pricing_sample.pct_change()[1:]
#plt.plot(returns_sample.index, returns_sample.values)
#plt.ylabel('Returns');
#
#
#test_statistic = (len(returns_sample) - 1) * returns_sample.std()**2 / 0.0001
#print ('Chi-square test statistic: ', test_statistic)
#
#
#crit_value = chi2.ppf(0.99, len(returns_sample) - 1)
#print ('Critical value at a = 0.01 with 251 df: ', crit_value)
#


#symbol_list = ["SPY", "AAPL"]
#start = "2015-01-01"
#end = "2016-01-01"
#pricing_sample = get_pricing(symbol_list, start_date = start, end_date = end, fields = 'price')
#pricing_sample.columns = map(lambda x: x.symbol, pricing_sample.columns)
#returns_sample = pricing_sample.pct_change()[1:]
#
#
#spy_std_dev, aapl_std_dev = returns_sample.std()
#print ('SPY standard deviation is: ', spy_std_dev)
#print ('AAPL standard deviation is: ', aapl_std_dev)


#
#
#test_statistic = (aapl_std_dev / spy_std_dev)**2
#print("F Test statistic: ", test_statistic)
#
#
#df1 = len(returns_sample['AAPL']) - 1
#df2 = len(returns_sample['SPY']) - 1
#
#print ('Degrees of freedom for SPY: ', df2)
#print ('Degrees of freedom for AAPL: ', df1)
#
#from scipy.stats import f
#
#
#upper_crit_value = f.ppf(0.975, df1, df2)
#lower_crit_value = f.ppf(0.025, df1, df2)
#print ('Upper critical value at a = 0.05 wi