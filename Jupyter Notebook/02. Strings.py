# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:02:03 2019

@author: JAINS
"""
#String variable types
print(3*'unm'+'ium') #concatenated strings
print('py' 'thon')
text = ('Put several strings within parentheses ' #concatenation using quotation marks
        'to have them joined together.')
prefix='py'
postfix='thon'
prefix+postfix #concatenated strings using variables

prefix[1] #indexing of strings
text[3:10] #characters in position 3 to 10 are displayed
text[:3]+text[7:9] #concatenation of strings using indexing
text1='J'+text[2:5] 

len(text) #returns the length of a string
len(text[3:23]) #returns the length of the selected portion of the string

