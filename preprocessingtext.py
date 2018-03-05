# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 17:27:12 2018

@author: pc
"""

text=open(r'C:\Users\pc\Desktop\input1.txt', 'r' , encoding='utf-8')
data=text.read().splitlines()
f=open(r'C:\Users\pc\Desktop\input.txt', 'w' , encoding='utf-8')
i=0
for line in data:
    for char in '<>-"'',[]@():':
        line=line.replace(char,'')        
    da=line.replace(".","\r\n")
    f.write(da)

character = input("Enter character: ")

with open(r'C:\Users\pc\Desktop\input.txt', 'r',encoding='utf-8') as f:
    print(sum(line.count(character) for line in f))