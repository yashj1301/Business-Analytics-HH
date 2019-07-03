#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:10:10 2019

@author: yash_j1301
"""

import pandas as pd
import numpy as np

df=pd.DataFrame(np.random.randint(30,100,size=(5,3)),columns=['HR','MM','Stats'], index=['Yash','Abhijeet','Saksham','Shashiraj','Beenu']) #creating a dataframe 
df
name=['Yash','Abhijeet','Shashiraj','Beenu','Nikhil','Hitesh','Priyank','Saksham','Dhiraj','Anil','Pooja'] 
df=df.reindex(name)  #adding new rows as indexes
df
df.isnull() #true values
df.notnull() #false values
df.isna()==df.isnull() #isna and isnull are the same

df['HR'].isnull().sum() #no of missing values
df['MM'].isnull().sum() #no of missing values

df['HR'].fillna(0,inplace=True) #filling the missing values in the column data
df['MM'].fillna(0,inplace=True)
df['Stats'].fillna(0,inplace=True)

df.fillna(0,inplace=True) #filling all missing values in the df
df.isnull().sum() #no of missing values from the dataframe

df.fillna(np.random.randint(10,20),inplace=True)
df1=df.fillna(method='pad') #copying values from row above 
df1
df2=df.fillna(method='backfill') #copying values from row below
df2
df.fillna(method='pad', axis=1) #copying values from columns

df3=df.dropna(axis=0) #eliminating the empty rows
df3

df4=df.dropna(axis=1) 
df4 #eliminating the empty columns

round(df.mean(axis=1),2) #mean marks
stats_mean=df['Stats'].mean()

df['HR'].fillna(stats_mean,inplace=True) #replacing empty values with mean
df['MM'].fillna(df['MM'].mean(), inplace=True)
df['Stats'].fillna(df['HR'].mean(),inplace=True)
df
df.mean(1).round(2)
df.mean().round(2)
