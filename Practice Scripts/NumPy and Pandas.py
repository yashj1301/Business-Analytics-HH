#Case Study - Denco

import pandas as pd
import numpy as np
pd.set_option('display.max_columns',25)

df = pd.read_csv('E:\\Henry Harvin\\PyProjects\\SIPPython\\pyAnalyticshhe\\data\\denco.csv')

df.head()
df.columns
len(df)
df.describe()
df.shape
df.dtypes
df['region'] = df['region'].astype('category')
df.dtypes

df.region.value_counts()
df.region.value_counts().plot(kind='bar')

df.columns

df.custname.value_counts()
df.custname.value_counts().sort_values(ascending=False)[1:10]

df.groupby('custname').size()

df.sort_values(['custname'])  
df.columns

df.groupby('custname').size().sort_values(ascending=False)
df.groupby('custname').size().sort_values(ascending=False).head(10)

df.groupby(['custname'])['margin'].nlargest(5)
df.sort_values(['revenue'], ascending= True).groupby( 'region' ).mean()

df.groupby('custname').aggregate({'revenue':np.sum})
df.groupby('custname').aggregate({'revenue':np.sum}).sort_values(by='revenue', ascending=False)
df.groupby('custname').aggregate({'revenue':np.sum}).sort_values(by='revenue', ascending=False).head(10)

df.groupby('custname').aggregate({'revenue':[np.sum, 'size'] }).sort_values(by='revenue', ascending=False) 
df[['revenue','custname']].groupby('custname').agg['size'].sort_values(by='revenue', ascending=False) 

df[['partnum','revenue']].sort_values(by='revenue', ascending=False)
df[['partnum','revenue']].sort_values(by='revenue', ascending=False).head(10)

df[['partnum','revenue']].groupby('partnum').sum()
df[['partnum','revenue']].groupby('partnum').sum().sort_values(by='revenue', ascending=False).head(10)

df[['partnum','margin']].sort_values(by='margin', ascending=False)
df[['partnum','margin']].sort_values(by='margin', ascending=False).head(10)

df[['partnum','margin']].groupby('partnum').sum()
df[['partnum','margin']].groupby('partnum').sum().sort_values(by='margin', ascending=False).head(10)

df.groupby('partnum').size().sort_values(ascending=False).head(10)

df[['revenue', 'region']].groupby( 'region').sum().sort_values(by='revenue', ascending=False)
df[['revenue', 'region']].groupby( 'region').sum().sort_values(by='revenue', ascending=False).plot(kind='bar')

