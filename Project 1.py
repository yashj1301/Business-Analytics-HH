#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 12:29:04 2019

@author: yash_j1301
"""

#Project 1


#1.Understanding the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline

cars_data=pd.read_csv('https://raw.githubusercontent.com/akjadon/Finalprojects_DS/master/Car_pricing_prediction/CarPrice_Assignment.csv')
cars_data.info()
cars_data.head()
cars_data.describe(percentiles=[0.25,0.5,0.75,1]).round(2)


#2. Data Cleaning and Preparation

#splitting the car name column and creating new columns  company and car model
cars_data=cars_data.join(cars_data['CarName'].str.split(' ',1,expand=True).rename(columns={0:'Company',1:'CarModel'}))
cars_data.head() 

#checking the columns created
cars_data['Company'].unique()
cars_data['CarModel'].unique()

#replacing incorrect values to correct values
cars_data['Company'].replace('maxda','mazda',inplace=True)
cars_data['Company'].replace(['vokswagen','vw'],'volkswagen',inplace=True)
cars_data['Company'].replace('porcshce','porsche',inplace=True)
cars_data['Company'].replace('toyouta','toyota',inplace=True)

#converting all the string data to lower to avoid any case difference errors
cars_data['Company']=cars_data['Company'].str.lower()

#checking for duplicate values
cars_data.loc[cars_data.duplicated()]

#checking columns
column_names=cars_data.columns.tolist()
column_names

#3. Visualizing the Data

plt.figure(figsize=(10,10)) #plot size according to scale 1 unit=72 pixels
plt.subplot(2,1,1) #2 rows, 1 column and index=1
plt.title('Car Price Distribution Plot') #title for the chart 
sns.distplot(cars_data['price']) #distplot for price of cars

plt.subplot(2,1,2)
plt.title('Car Price Spread')
sns.boxplot(cars_data['price']) #distribution of price in the data

"""

    Categorical Data 

    - Company
    - Symboling
    - fueltype
    - enginetype
    - carbody
    - doornumber
    - enginelocation
    - fuelsystem
    - cylindernumber
    - aspiration
    - drivewheel

"""
#1. Car Company

plt1 = cars_data['Company'].value_counts().plot('bar')
plt.title('Companies Histogram')
plt1.set(xlabel = 'Car Company', ylabel='Frequency of Company')

#2. Fuel Type

plt.figure(figsize=(25, 6))
plt.title('Fuel Type Chart')
labels=cars_data['fueltype'].unique()
plt2 = cars_data['fueltype'].value_counts().tolist()
plt.pie(plt2,labels=plt2, autopct='%1.1f%%')
plt.legend(labels)

#3. Car Body Type
plt.figure(figsize=(15,8))
plt.title('Car Body Type Chart')
labels=cars_data['carbody'].unique()
plt3 = cars_data['carbody'].value_counts().tolist()
plt.pie(plt3, labels=plt3, autopct='%1.1f%%')
plt.legend(labels, loc=1)

plt.show()

