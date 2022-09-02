# -*- coding: utf-8 -*-

"Spyder Editor"

"This is a temporary script file."

"Welcome to Python"

#Variables

x=4 #int variable
y=5.6 #float variable
z=3+5j #complex variable
abc='abc' #string literal

#Conversions

y_int=int(y) #float to int
x_float=float(x) #int to float


x=str(x) #int to string
y=str(y) #float to string
z=str(z) #complex to string

#Operators

   #1. Arithmetic Operators

          p=x+y            #int and float = float
          q=x+y+z          #float and complex = complex
          r=x+z            #int and complex = complex
         
   #2. Assignment Operators

          a=4        #assigning the value of 4 to variable 'a'
          a+=7       #a=a+7 - increment operator
          a-=3       #a=a-3 - decrement operator 
          a*=2       #a=a*2 - multiplication operator
          a%=7       #a=a%7 - modulus operator(remainder)
          a**=4      #a=a**4 - power operator   
          a/=3       #a=a/3 - division operator; returns the precise value
          a//=2      #a=a//2 - floor division operator; returns the rounded off value
          a&=4       #a=a&3 - returns 0 if false and the value if true (AND Operator)
          a^=12      #a=a^12 - returns 0 if true, and adds the value if false (XOR Operator)
          a>>=13     #a=a>>13 - returns 0 if true, 1 if false
          
   #3. Comparison Operators

          b==6   #equals to operator - returns true if equal, false if not
          a!=b   #not equals to operator - returns true if not equal, false if equal         
          a>=b   #greater than or equals to operator - returns true if LHS is equal to or greater than RHS
          a<=b   #less than or equals to operator - returns true if LHS is equal to or less than RHS
          a>b    #greater than operator - returns true if LHS is greater than RHS
          a<b    #less than operator - returns true if LHS is less than RHS
          
   #4. Logical Operators
          
          if (a+b)>0 and (a-b>0): print(True)
          else:  print(False)                  #and operator - returns true if both conditions are true     
          
          if(a+b-3)>=0 or (a-b+3)<0: print(True)
          else: print(False)                  #or operator - returns true if one of the conditions is true  
          
          not(a+b==10) #not operator - returns false if the condition is true and vica versa
          
   #5. Identity Operators
    
          list_1=[1,2,3,1]
          list_2=list_1
          print(list_1 is list_2) #is operator - shows true if two literals share the same memory location
          print(list_1 is not list_2) #not is operator - shows true when two literals do not share the same memory location

   #6. Membership Operators 
   
          print(3 in list_1) #in operator - returns true if the value is found in the sequence 
          print(17 in list_1) #not in operator - returns true if the value is not found in the sequence
