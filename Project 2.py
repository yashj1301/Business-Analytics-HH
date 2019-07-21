#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:26:37 2019

@author: yash_j1301
"""
#-----------------------------------------------------------------------------------------------------------------------#
#                                       Project 2 - Human Resource (HR) Analytics 
#-----------------------------------------------------------------------------------------------------------------------#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
%matplotlib inline

#1. Data Understanding

hr_data=pd.read_csv('//home//yash_j1301//Downloads//HR_Processed_data.csv')
hr_data.head()

hr_data.columns #column names
hr_data.shape[0] #no of entries
hr_data.shape[1] #no of columns

hr_data.info()
hr_data.describe(percentiles=[0.25,0.5,0.75,1]).round(2)


#2. Data Cleaning and Preparation

hr_data.columns
hr_data=hr_data.rename(columns={'average_montly_hours':'average_monthly_hours'}) #renaming incorrect values
hr_data.columns=hr_data.columns.str.lower() #changing the case to lower-case

#checking for duplicate values
hr_data.loc[hr_data.duplicated()] #checking duplicate values
hr_data.drop_duplicates(keep=False, inplace=True) #deleting duplicate values


#3. Visualizing the Data

plt.figure(figsize=(10,15))
plt.subplot(1,3,1)
hr_data.groupby(['department'])['left'].value_counts() #checking how many from each department left and didn't leave
sns.countplot(x = 'department', hue = 'left', data = hr_data)
plt.subplot(1,3,2)
hr_data['department'].value_counts().plot('bar') #people in each department
plt.subplot(1,3,3)
sns.countplot(x = 'salary', hue='left', data = hr_data) #salary of people leaving and not leaving

hr=hr_data.copy()
hr['salary'] = hr['salary'].astype('category')
hr['salary'] = hr['salary'].cat.reorder_categories(['low', 'medium', 'high'])
hr['salary'] = hr['salary'].cat.codes

sns.pointplot(x='department', y='salary', hue='left', data=hr) #trend 

#combining support, technical and IT into one value - technical
hr_data['department']=np.where(hr_data['department'] =='support', 'technical', hr_data['department']) 
hr_data['department']=np.where(hr_data['department'] =='IT', 'technical', hr_data['department'])

hr_data['department'].unique() #checking the unique values

#exploring categorical columns
hr_data.groupby('left').mean()
hr_data.groupby('department').mean()
hr_data.groupby('salary').mean()

#plot - employee turnover of the company
pd.crosstab(hr_data['department'],hr_data['left']).plot(kind='bar')
plt.title('Turnover Frequency for Department')
plt.xlabel('Department')
plt.ylabel('Frequency of Turnover')

#numeric data

"""
Categorical Variables

- Salary
- Department
- Number of Projects
- Left
- Promotion last 5 years
- Work Accident
- Time Spent in Company

"""


#Dummy Variables

def dummies(x,df):
    var=pd.get_dummies(df[x], drop_first=True)
    df=pd.concat([df,var], axis=1)
    df.drop([x], axis=1, inplace=True)
    return df

hr_data=dummies('department',hr_data)
hr_data=dummies('salary',hr_data)
hr_data=dummies('number_project',hr_data)
hr_data=dummies('promotion_last_5years',hr_data)
hr_data=dummies('work_accident',hr_data)
hr_data=dummies('time_spend_company',hr_data)

#creating the target and independent variable
hr_var=hr_data.columns.tolist()
y=['left']
x=[var for var in hr_var if var not in y]

#Starting Regression

from sklearn.feature_selection import RFE #Recursive Feature Selection for Selecting Features
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, 15)
rfe = rfe.fit(hr_data[x], hr_data[y])
print(rfe.support_)
print(rfe.ranking_)

list(zip(hr_data[x].columns,rfe.support_,rfe.ranking_))

num_vars=hr_data[x].columns[rfe.support_] #selected features

x=hr_data[num_vars]
y=hr_data['left']

#creating regression model

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

y_pred=logreg.predict(x_test)
from sklearn.metrics import accuracy_score
print('Logistic regression accuracy: ',(accuracy_score(y_test,y_pred )*100).round(3),'%') #accuracy score of the model

y_pred=pd.DataFrame(y_pred)

y_test=pd.DataFrame(y_test)
y_test=y_test.reset_index(drop=True)

p=pd.concat([y_test, y_pred], axis=1)
p=p.rename(columns={0:'pred_left'})

#random forest 
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)

rf_pred=pd.DataFrame(rf.predict(x_test))
acc=accuracy_score(y_test,rf_pred)*100
print('Random Forest accuracy: ',acc.round(3),'%') #accuracy score of the model

rf_pred[0].value_counts()
rf_pred.index

#changing to dataframe for plotting

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Actual left')
plt.xlabel('Index')
plt.ylabel('Frequency')
p['left'].value_counts().plot('bar')
plt.subplot(1,2,2)
plt.title('Predicted left')
plt.xlabel('Index')
plt.ylabel('Frequency')
p['pred_left'].value_counts().plot('bar')
plt.show()

#validation techniques - confusion matrix for logistic regression

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

cm_rf=confusion_matrix(y_test,rf_pred)
cm_rf

#validation techniques - K-fold for random forest
from sklearn import model_selection
from sklearn.model_selection import cross_val_score,KFold
kfold = KFold(n_splits=15, random_state=100)
modelCV = RandomForestClassifier()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, x_test, y_test, cv=kfold, scoring=scoring)
print("15-fold cross validation average accuracy: ",((results.mean())*100).round(3),'%')

#classification report to check generalisation of data
from sklearn.metrics import classification_report
print(classification_report(y_test, rf_pred)) #random forest report
print(classification_report(y_test, y_pred)) #logistic regression report
 
a=rf.predict_proba(x_test) #predicted probability
a=pd.DataFrame(a)
a.loc[:,0] #probability for 0 - didn't leave 
a.loc[:,1] #probability for 1 - left  


#roc curve
from sklearn.metrics import roc_curve, roc_auc_score
fpr,tpr,threshold=roc_curve(y_test,y_pred)
fpr_rf,tpr_rf,threshold_rf=roc_curve(y_test,rf_pred)

roc_auc=roc_auc_score(y_test,y_pred)
roc_auc_rf=roc_auc_score(y_test,rf_pred)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr,'b',label='Logistic Regression (area=%0.2f)'%roc_auc)
plt.plot(fpr_rf,tpr_rf,'g',label='Random Forest (area=%0.2f)'%roc_auc_rf)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------#
#                                                        Project End
#-----------------------------------------------------------------------------------------------------------------------#


