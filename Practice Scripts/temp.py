# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file
"""
from pydataset import data #importing dataset
import numpy
import pandas as pd
mt_cars=data('mtcars') #defining the name to another dataset
mt_cars

mt_cars.describe() #summary
mt_cars.columns #column names
mt_cars.shape[1] #no of colums

columns=mt_cars.columns.astype('category')
columns

len(dir(pd))
len(dir(numpy))

pd2=pd.read_clipboard()
pd2
pd2.to_clipboard()
pd2[1:3]
pd2.iloc[1:9,3:6]
pd2.loc[1:4,['OrderDate','Region']]
pd2
