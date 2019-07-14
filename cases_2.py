#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:56:52 2019

@author: yash_j1301
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

classDate = pd.date_range('2019-06-15', '2019-07-28', freq='B')
classDate
type(classDate)
len(classDate)

classDate.strftime('%a')  

random.seed(1234)
attendance1 = np.random.randint(low=25,high=50, size=30)
attendance1 
np.mean(attendance1)

workshop1 = pd.DataFrame({'Attendance':attendance1, 'Days' : classDate.strftime( '%a') }, index=classDate)
workshop1
workshop1.head(3).append(workshop1.tail(2))
workshop1.describe().round(2)
workshop1.dtypes

workshop1['Days'] = workshop1['Days'].astype('category', ordered=True)
workshop1['Days'] #warning - Friday before Monday

from pandas.api.types import CategoricalDtype
Days = CategoricalDtype(categories=['Mon', 'Tue', 'Wed', 'Thu','Fri'], ordered=True)
workshop1['Days'] = workshop1['Days'].astype(Days)
workshop1['Days']

workshop1['Attendance'].plot(figsize=(10,5)) 
#only on those days where attendance was taken

workshop1.asfreq('D').plot(figsize=(10,5)) #gaps between graphs because of sat and sunday

#fill these gaps : only 1,2 ahead ffill
workshop1.asfreq('D')['Attendance'].fillna(method='ffill', limit=2).plot(figsize=(10,5))


#average attendance on Mondays, Tues,
workshop1.groupby(['Days', pd.Grouper(freq='M')])['Attendance'].mean().loc[['Mon','Tue']]

#all mondays on month; find their mean day wise
workshop1.groupby(['Days', pd.Grouper(freq='W')])['Attendance'].mean()

#all mondays on week; find their mean day wise
workshop1.groupby(['Days', pd.Grouper(freq='D')])['Attendance'].mean()

#same as original data - full year
workshop1.groupby(['Days', pd.Grouper(freq='A')])['Attendance'].mean()
workshop1.groupby('Days').mean().round(3)
workshop1

#Moving average
workshop1.head(5)
(workshop1['Attendance'][0:3].sum()/3).round(3)
   
(40+49+37)/3, (49+37+39)/3
workshop1['Attendance'].rolling(3).mean().head(5).fillna(workshop1['Attendance'].mean()).round(2)

ma1 = workshop1.rolling(5, win_type='triang').mean().fillna(workshop1['Attendance'].mean()).round(2)
ma1  #moving average over 5 day period

#plot it
ma1.plot(color='blue', linewidth=2.5, marker='', figsize=(10,5), label='Moving Average Attenance 5 Day Period')

#using MPL 
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(15,4))
ma1.plot(ax=ax)

workshop1['Attendance'].plot(ax=ax)
xs = workshop1.index
ys = workshop1['Attendance']
y1s = ma1['Attendance']
y1s

for x,y in zip(xs,ys):
    label = "{:d}".format(y)
    plt.annotate(label, (x,y), textcoords="offset points",xytext =(3,2), ha='center', color ='red', size=12) 
    
for x,y in zip(xs,y1s):
    plt.annotate("{:.0f}".format(y), (x,y), color='blue', size=10) 
fig.text(0.7, 0.8, r'Attendance Rolling Plot', ha="center", va="bottom", size="medium",color="blue")
ax.set_title("Time Series Plot")
plt.show();


fig, ax = plt.subplots(figsize=(15,4))
workshop1['Attendance'].plot(ax=ax)
myFmt = DateFormatter("%d/%a")
ax.xaxis.set_major_formatter(myFmt)
myxticks = pd.date_range(workshop1.index.min(), workshop1.index.max(), freq='D')
plt.xticks(myxticks, rotation='vertical') 
plt.show()




