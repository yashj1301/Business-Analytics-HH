# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:57:05 2019

@author: JAINS
"""
#for loops

words=['cat','window','defenestrate']

for w in words: print(w,len(w),end=' ')

for w in words[1:7]:
    if len(w)>3:
        words.insert(5,'19 Jun 2019')
        
print(words)         #using for loop to add and repeat words using if else
len(w)
        
for i in range(5): 
    print(words[i], end=' ') #using range function in for loop
               
for i in range(5,10) : print(words[i], end=' ') #defining the range in for loop
for i in range(3,20,4) : print(i, end=' ') #creating an arithmetic progression (AP)

#Break and Continue Statements
for i in range(2,10): 
    for x in range(2,i): 
        if i%x==0:
            print(i,'=',x,'x',i//x)
            break #breaking the loop to avoid multiple display of values
        else: 
            print(i,' is a prime number.')
            break # breaking the else loop
        
for num in range(2, 10):
     if num % 2 == 0:
         print("Found an even number", num)
         continue #continuing the for loop after running the if loop
     print("Found a number", num)        
     
world_cup=['India','Australia','England','South Africa','Pakistan','New Zealand','Sri Lanka','Afghanistan','West Indies']

for i in range(1,len(world_cup),2): print(world_cup[i],sep=',', end=' ')
for i in range(0,len(world_cup),2): print(world_cup[i],sep=',', end=' ')
    