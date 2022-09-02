# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 21:27:48 2019

@author: JAINS
"""

thistuple=('apple','banana','cherry','potato','tomato','ladyfinger') #defining a tuple
thistuple #printing the tuple

thistuple[2] #indexing the tuple
thistuple[2:4]
thistuple[:3]

tuple1=('gdfgdfg','dfgdfdf','fhfdhggfn','fhfdhdfhcc','dsfdsgfxbb','fhfdhsvsdgv')
tuple2=tuple1+thistuple #concatenation of tuples
tuple2

tuple2[4]='yash' #error - immutability of tuples
tuple2[3:6]=[] #error

tuple[:]=[] #error 
len(tuple2) #no of items in the tuple

#nested tuples
tuple3=('yash','jain',tuple1,'yj','samosa',thistuple) 
tuple3[2] #tuple inside a tuple
tuple3[2][4] #accessing tuple elements

list=['python','anaconda','spyder','IDE']
tuple4=(tuple3,list,'user') #list inside a tuple
tuple4

#taking tuple elements from user by taking list as input and converting the list to tuple
list1=[]
n=int(input('Enter the no of elements: '))
for i in range(0,n) : 
    print('Enter the element ',i+1,': ')
    ele=int(input())
    list1.append(ele)
    
list1=tuple(list1)    
    
    

    