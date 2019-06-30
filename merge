#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 16:04:26 2019

@author: yash_j1301
"""

#merging two dataframes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pydataset import data

mtcars=data('mtcars')

df1=pd.DataFrame({'roll no': ['SO1','SO2','SO3'], 'marks1': [55,66,75]})
df2=pd.DataFrame({'roll no': ['SO1','SO2','SO3'], 'marks2': [56,76,89]})

merged_df=df1.merge(df2,left_on='roll no',right_on='roll no')
merged_df #merging dataframe

df1.set_index('roll no',inplace=True)
df2.set_index('roll no',inplace=True)

merged_df1=pd.merge(df1, df2, on='roll no', how='outer') #outer merge
pd.merge(df1, df2, on='roll no', how='inner') #inner merge
pd.merge(df1, df2, on='roll no', how='left') #left merge
pd.merge(df1, df2, on='roll no', how='right') #right merge
pd.merge(df1, df2, on='roll no', how='outer', left_index=False) #outer merge with indexing
pd.merge(df1, df2, on='roll no', how='outer', left_index=True) #outer merge

x=mtcars['cyl'].value_counts()
x
x.plot(kind='bar')
