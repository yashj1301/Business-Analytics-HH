#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 12:32:24 2019

@author: yash_j1301
"""
#Multi-indexing 
import numpy as np
import matplotlib.pyplot as plt
from pydataset import data

mtcars=data('mtcars')
mtcars.head(10)
dir(data)
iris=data('iris')
titanic=data('titanic')
titanic.head(100)

data1=mtcars.copy()
data1.columns #column names
data1.values #matrix of numerical values 
data1.index #unique rows
data1.am.dtypes #data types

data1[['am','cyl','mpg','hp','drat','disp','wt','qsec','vs','gear','carb']].astype('category') #converting list to category
data1.dtypes #data types of the dataframe

data2=data1.reset_index() #resetting the previously set indexes of the dataframe
data2.index
data2.iloc[0:3,0:4] #index column created
data2.rename({'index':'carname'},inplace=True,axis='columns')
data2.columns[0] #column name renamed to carname
data3cyl=data2.set_index('cyl')
data3cyl.index 
data3cyl.head()
data3cyl.iloc[[1],[3]] #first entry with third column
data3cyl.am.value_counts() #no of occurrences of each unique value
data3cyl.loc[4] #all the entries with cyl=4
data3cyl.columns
pd.set_option('display.max_columns',20)
data3=data2.set_index('gear',drop=True).head()
data3.head(20)
data3.reset_index().set_index('cyl').head()
data3.set_index('cyl',append=True).set_index('am',append=True) #multiple indexing
data_group_by_onecol=data2.groupby(['gear']).mpg.mean() #gear-wise mean mileage 
type(data_group_by_onecol)

data_group_by_twocol=data2.groupby(['gear','am']).mpg.mean() #gear-wise mean mileage of auto tx 
data_group_by_twocol.index #multi-level indexing


