# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file
"""
from pydataset import data #importing dataset
import numpy; import pandas
mt_cars=data('mtcars') #defining the name to another dataset
mt_cars

mt_cars.describe() #summary
mt_cars.columns #column names
mt_cars.shape[1] #no of colums

columns=mt_cars.columns.astype('category')
columns

len(dir(pandas))
len(dir(numpy))
