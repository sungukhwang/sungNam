# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:28:12 2018

@author: pc
"""


import os
path=r'C:/Users/pc/Desktop/shm/tacotron/datasets/yuinna/audio/'
a=list(range(830))
print(a)
i=0
for fname in os.listdir(path):
    fullpath = path + fname
    print(fullpath)
    fpre=fname[:6]
    fpost=fname[6:]

    if fpost=='.wav':
        fpre2=str(a[i])   
        rename=path+ fpre2 + fpost
        print(rename)
        i=i+1
        os.rename(fullpath,rename)
        
