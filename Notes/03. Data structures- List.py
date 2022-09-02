# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:25:02 2019

@author: JAINS
"""

squares=[1,4,9,16,25] #defining a list
squares
 
squares[3] #indexing of lists
squares[1:4]
squares[:5]

rectangles=[3,6,8,3,5,2,7]
array=squares+rectangles   #concatenating lists
array 

array[5]=34  #mutability of lists
array 
array[3:6]=[] #deleting the values
array[:]=['yash','shashiraj','saksham','abhijeet'] #changing the list from int to str
array[1:3]=[1,'abc',45] #replacing the values 

squares[:]=[] #emptying the list
len(array) #no of elements in the list

#Nested Listing

squares[:]=[1,5,4,6,3,rectangles]
array[:]=[1,4,'yash',87,'jain',squares,rectangles]
array[5] #list inside a list
array[5][5] #accessing the list through indexing
array[5][5][5] 

#Accepting values from user
list=[] 
n=int(input('Enter the number of elements: '))
for i in range(0,n): 
 ele=int(input()) 
 list.append(ele)

    
print('The list is: ',list)
