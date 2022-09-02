#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:16:53 2019

@author: yash_j1301
"""

import datetime #date & time library
print(dir(datetime))
print(dir(datetime.timedelta),end=',')

dt=datetime.datetime.now() #current date and time from system
print(dt)

dt.min #minimum value for date available in python
dt.max #maximum value for date available in python

print(dt.year) #current year
print(dt.strftime('%A')) #day from date
date_1=datetime.datetime(2019,7,13,13,53,45) #declaring a variable in a date format
print(date_1)

print(date_1.day) #day
print(date_1.month) #month
print(date_1.year) #year

from datetime import datetime as dt
today=dt.now() #storing today's date
today

dt.strftime(today,'%d/%m/%y') #declaring a format for yourself
date=dt.strptime('3/9/2019','%d/%m/%Y') #declaring a string as a date
date.weekday() #day of the week


diff=date-date_1 #difference between two dates
print(diff)

today=dt.now()
birth=dt(2000,7,13,15,6,34)
age=today-birth #age in days

birth.hour #hour
birth.minute #minute
birth.second #second
birth=birth.replace(day=12) #replacing day
birth=birth.replace(day=10, month=3) #replacing day and month
birth=birth.replace(day=13, month=7, year=2000) #replacing day,month and year

today=datetime.datetime.now()
print(today+datetime.timedelta(days=45)) #date + no of days

