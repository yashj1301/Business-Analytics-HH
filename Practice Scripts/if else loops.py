# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:21:36 2019

@author: JAINS
"""
x=int(input('Please enter the number: '))
if x<0:
    x=0
    print('Negative changed to zero')
    
elif x==0: print('Zero')    
elif x==1: print('Single')
else : print('More')