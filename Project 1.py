#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 12:29:04 2019

@author: yash_j1301
"""

#Project 1 - Car Price Prediction


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
plt.show()

"""

    Categorical Data 

    - Company               #
    - Symboling             #
    - fueltype              #
    - enginetype            #
    - carbody               #
    - doornumber            #
    - enginelocation        #
    - fuelsystem            #
    - cylindernumber        #
    - aspiration            #
    - drivewheel            #


"""
#1. Car Company

plt.figure(figsize=(30, 15))

#plot 1.1
plt.subplot(1,2,1)
plt1 = cars_data['Company'].value_counts().plot('bar')
plt.title('Companies Histogram')
plt1.set(xlabel = 'Car Company', ylabel='Frequency of Company')
xs=cars_data['Company'].unique()
ys=cars_data['Company'].value_counts()
plt.bar(xs,ys)
for x,y in zip(xs,ys):
    label = "{:.2f}".format(y)
    plt.annotate(label,(x,y), textcoords="offset points",xytext=(5,5),ha='center') 
plt.xticks(xs)

#plot 1.2
plt.subplot(1,2,2)
company_vs_price = pd.DataFrame(cars_data.groupby(['Company'])['price'].mean().sort_values(ascending = False))
plt2=company_vs_price.index.value_counts().plot('bar')
plt.title('Company Name vs Average Price')
plt2.set(xlabel='Car Company', ylabel='Average Price')
xs=company_vs_price.index
ys=company_vs_price['price'].round(2)
plt.bar(xs,ys)
for x,y in zip(xs,ys):
    label = "{:.2f}".format(y)
    plt.annotate(label,(x,y), textcoords="offset points",xytext=(5,5),ha='center') 
plt.xticks(xs)
plt.tight_layout()
plt.show()


#2. Fuel Type

plt.figure(figsize=(25, 6))

#plot 2.1
plt.subplot(1,2,1)
plt.title('Fuel Type Chart')
labels=cars_data['fueltype'].unique()
plt3 = cars_data['fueltype'].value_counts().tolist()
plt.pie(plt3,labels=plt3, autopct='%1.1f%%')
plt.legend(labels)

#plot 2.2
plt.subplot(1,2,2)
fuel_vs_price = pd.DataFrame(cars_data.groupby(['fueltype'])['price'].mean().sort_values(ascending = False))
plt4=fuel_vs_price.index.value_counts().plot('bar')
plt.title('Fuel Type vs Average Price')
plt4.set(xlabel='Fuel Type', ylabel='Average Price')
xs=fuel_vs_price.index
ys=fuel_vs_price['price'].round(2)
plt.bar(xs,ys)
for x,y in zip(xs,ys):
    label = "{:.2f}".format(y)
    plt.annotate(label,(x,y), textcoords="offset points",xytext=(5,5),ha='center') 
plt.xticks(xs)
plt.tight_layout()
plt.show()


#3. Car Body Type

plt.figure(figsize=(15,10))

#plot 1
plt.subplot(1,2,1)
plt.title('Car Body Type Chart')
labels=cars_data['carbody'].unique()
plt5 = cars_data['carbody'].value_counts().tolist()
plt.pie(plt5, labels=plt5, autopct='%1.1f%%')
plt.legend(labels, loc=1)

#plot 2
plt.subplot(1,2,2)
car_vs_price = pd.DataFrame(cars_data.groupby(['carbody'])['price'].mean().sort_values(ascending = False))
plt6=car_vs_price.index.value_counts().plot('bar')
plt.title('Car Body Type vs Average Price')
plt6.set(xlabel='Car Body Type', ylabel='Average Price')
xs=car_vs_price.index
ys=car_vs_price['price'].round(2)
plt.bar(xs,ys)
for x,y in zip(xs,ys):
    label = "{:.2f}".format(y)
    plt.annotate(label,(x,y), textcoords="offset points",xytext=(0,2),ha='center') 
plt.xticks(xs)
plt.show()


#4. Symboling

plt.figure(figsize=(25,10))

#plot 1
plt.subplot(1,2,1)
plt.title('Symboling Chart')
labels=cars_data['symboling'].unique()
plt7 = cars_data['symboling'].value_counts().tolist()
plt.pie(plt7, labels=plt7, autopct='%1.1f%%')
plt.legend(labels, loc=1)

#plot 2
plt.subplot(1,2,2)
plt.title('Symboling vs Price')
sns.boxplot(x=cars_data['symboling'], y=cars_data['price'])
plt.show()


#5. Engine Type

plt.figure(figsize=(25,10))

#plot 1
plt.subplot(1,2,1)
plt8 = cars_data['enginetype'].value_counts().plot('bar')
plt.title('Engine Type Histogram')
plt8.set(xlabel = 'Engine Type', ylabel='Frequency')
xs=cars_data['enginetype'].unique()
ys=cars_data['enginetype'].value_counts()
plt.bar(xs,ys)
for x,y in zip(xs,ys):
    label = "{:.2f}".format(y)
    plt.annotate(label,(x,y), textcoords="offset points",xytext=(0,2),ha='center') 
plt.xticks(xs)

#plot 2
plt.subplot(1,2,2)
plt.title('Engine Type vs Price')
sns.boxplot(x=cars_data['enginetype'], y=cars_data['price'])
plt.show()


#6. Door Number 
 

plt.figure(figsize=(25,10))

#plot 1
plt.subplot(1,2,1)
labels=cars_data['doornumber'].unique()
plt8 = cars_data['doornumber'].value_counts().tolist()
plt.title('No of Doors Chart')
plt.pie(plt8, labels=plt8, autopct='%1.1f%%')
plt.legend(labels, loc=1)

#plot 2
plt.subplot(1,2,2)
plt.title('No of Doors vs Price')
sns.boxplot(x=cars_data['doornumber'], y=cars_data['price'])
plt.show()


#7. Engine Location

plt.figure(figsize=(25,10))

#plot 1
plt.subplot(1,2,1)
labels=cars_data['enginelocation'].unique()
plt9 = cars_data['enginelocation'].value_counts().tolist()
plt.title('Engine Location Chart')
plt.pie(plt9, labels=plt9, autopct='%1.1f%%')
plt.legend(labels, loc=1)

#plot 2
plt.subplot(1,2,2)
plt.title('Engine Location vs Price')
sns.boxplot(x=cars_data['enginelocation'], y=cars_data['price'])
plt.show()


#8. Fuel System

plt.figure(figsize=(25,10))

#plot 1
plt.subplot(1,2,1)
plt10 = cars_data['fuelsystem'].value_counts().plot('bar')
plt.title('Fuel System Type Histogram')
plt10.set(xlabel = 'Fuel System Type', ylabel='Frequency')
xs=cars_data['fuelsystem'].unique()
ys=cars_data['fuelsystem'].value_counts()
plt.bar(xs,ys)
for x,y in zip(xs,ys):
    label = "{:.2f}".format(y)
    plt.annotate(label,(x,y), textcoords="offset points",xytext=(0,2),ha='center') 
plt.xticks(xs)

#plot 2
plt.subplot(1,2,2)
plt.title('Fuel System Type vs Price')
sns.boxplot(x=cars_data['fuelsystem'], y=cars_data['price'])
plt.show()


#9. Cylinder Number '

plt.figure(figsize=(25,10))

#plot 1
plt.subplot(1,2,1)
plt11 = cars_data['cylindernumber'].value_counts().plot('bar')
plt.title('Cylinder Number Histogram')
plt11.set(xlabel = 'Cylinder Number', ylabel='Frequency')
xs=cars_data['cylindernumber'].unique()
ys=cars_data['cylindernumber'].value_counts()
plt.bar(xs,ys)
for x,y in zip(xs,ys):
    label = "{:.2f}".format(y)
    plt.annotate(label,(x,y), textcoords="offset points",xytext=(0,2),ha='center') 
plt.xticks(xs)

#plot 2
plt.subplot(1,2,2)
plt.title('Cylinder Number vs Price')
sns.boxplot(x=cars_data['cylindernumber'], y=cars_data['price'])
plt.show()


#10. Aspiration

plt.figure(figsize=(15,5))

#plot 1
plt.subplot(1,2,1)
labels=cars_data['aspiration'].unique()
plt12 = cars_data['aspiration'].value_counts().tolist()
plt.title('Aspiration Type Chart')
plt.pie(plt12, labels=plt12, autopct='%1.1f%%')
plt.legend(labels, loc=1)

#plot 2
plt.subplot(1,2,2)
plt.title('Engine Location vs Price')
sns.boxplot(x=cars_data['aspiration'], y=cars_data['price'])
plt.show()


#11. Drivewheel

plt.figure(figsize=(15,5))

#plot 1
plt.subplot(1,2,1)
labels=cars_data['drivewheel'].unique()
plt13 = cars_data['drivewheel'].value_counts().tolist()
plt.title('Drive Wheel Chart')
plt.pie(plt13, labels=plt13, autopct='%1.1f%%')
plt.legend(labels, loc=1)

#plot 2
plt.subplot(1,2,2)
plt.title('Drive Wheel vs Price')
sns.boxplot(x=cars_data['drivewheel'], y=cars_data['price'])
plt.show()


"""

Numerical Variables 

-Car Length                             #
-Car Width                              #
-Car Height                             #
-Curb Weight                            #
-Horsepower                             #
-Bore Ratio                             #
-Compression Ratio                      #
-Highway miles per gallon (mpg)         #        
-Engine Size                            #                  
-Stroke                                 #
-City Miles per gallon (mpg)            #
-Peak Revolutions per Minute (rpm)      #
-Wheel Base                             #
 

"""

def scatterplot(df,var):
    
    plt.scatter(df[var],df['price'])
    plt.xlabel(var); plt.ylabel('Price')
    plt.title('Scatter Plot for '+var+' vs Price')
    

#1. Car Length, Width and Height

    plt.figure(figsize=(15,20))
    plt.subplot(4,2,1)
    scatterplot(cars_data,'carlength')    
    plt.subplot(4,2,2)
    scatterplot(cars_data,'carwidth')
    plt.subplot(4,2,3)
    scatterplot(cars_data,'carheight')
    plt.show()
    plt.tight_layout()
    
#2. Creating a new variable- Car Volume

    cars_data['carvolume']=cars_data['carlength']*cars_data['carwidth']*cars_data['carheight']    
    cars_data['carvolume'].unique()
    scatterplot(cars_data,'carvolume')

#3. Curb Weight (Effective Weight of Car including its internal components), HorsePower, Boreratio, and Compression Ratio
    
    plt.figure(figsize=(15,20))
    plt.subplot(4,2,1)
    scatterplot(cars_data,'curbweight')    
    plt.subplot(4,2,2)
    scatterplot(cars_data,'horsepower')
    plt.subplot(4,2,3)
    scatterplot(cars_data,'boreratio')
    plt.subplot(4,2,4)
    scatterplot(cars_data,'compressionratio')
    plt.show()
    plt.tight_layout()    
    
#4. Creating a new Variable - Fuel Economy

    cars_data['fueleconomy']=(cars_data['citympg']*0.55)+(cars_data['highwaympg']*0.45)      
    cars_data['fueleconomy'].unique()
    scatterplot(cars_data,'fueleconomy')

#5. Creating a Categorical Variable - Car Class
    
    price=cars_data['price'].tolist()
    price
    carsrange=[]
    for i in cars_data['price']:
        if (i>0 and i<9000): carsrange.append('Low')
        elif (i>9000 and i<18000): carsrange.append('Medium-Low')
        elif (i>18000 and i<27000): carsrange.append('Medium')
        elif(i>27000 and i<36000): carsrange.append('High-Medium')
        else : carsrange.append('High')
    cars_data['carsrange']=carsrange
    cars_data['carsrange'].unique()

#plot
plt.figure(figsize=(5,5))
plt14 = cars_data['carsrange'].value_counts().plot('bar')
plt.title('Cars Range Histogram')
plt14.set(xlabel = 'Car Range', ylabel='Frequency')
xs=cars_data['carsrange'].unique()
ys=cars_data['carsrange'].value_counts()
plt.bar(xs,ys)
for x,y in zip(xs,ys):
    label = "{:.2f}".format(y)
    plt.annotate(label,(x,y), textcoords="offset points",xytext=(0,2),ha='center') 
plt.xticks(xs)

 
#6. Highway mpg and City mpg 

sns.pairplot(cars_data, x_vars=['highwaympg','citympg'], y_vars='price', height=4, aspect=1, kind='scatter')   

#7. Bore Ratio and Compression Ratio

sns.pairplot(cars_data, x_vars=['boreratio','compressionratio'], y_vars='price', height=4, aspect=1, kind='scatter')   

#8. Engine Size, Stroke, RPM and Wheelbase
        
    plt.figure(figsize=(15,20))
    plt.subplot(4,2,1)
    scatterplot(cars_data,'enginesize')    
    plt.subplot(4,2,2)
    scatterplot(cars_data,'stroke')
    plt.subplot(4,2,3)
    scatterplot(cars_data,'peakrpm')
    plt.subplot(4,2,4)
    scatterplot(cars_data,'wheelbase')
    plt.show()
    plt.tight_layout()    

#Correlation with price(target variable)
    
corr=cars_data.corr().round(3).loc['price']
corr=pd.DataFrame(corr)
corr
result=[]
    for i in corr['price']:
        if (i>-1 and i<-0.4): result.append('strong negative ')
        elif (i>-0.4 and i<-0.2): result.append('moderate negative ')
        elif (i>-0.2 and i<0): result.append('weak negative ')
        elif(i>0 and i<0.2): result.append('weak positive')
        elif(i>0.2 and i<0.5): result.append('moderate positive')
        else : result.append('strong positive')
    
corr['correlation']=result
corr['correlation'].value_counts()

plt.figure(figsize=(10,10))
plt.title('Correlation Chart')
labels=corr['correlation'].unique()
plt15 = corr['correlation'].value_counts().tolist()
plt.pie(plt15, labels=plt15, autopct='%1.1f%%')
plt.legend(labels, loc=1)


