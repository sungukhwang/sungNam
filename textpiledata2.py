# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 19:38:18 2018

@author: pc
"""


import codecs
import re
import collections
import requests
import time
from bs4 import BeautifulSoup
import string
import urllib
import urllib.request
import urllib.parse
import webbrowser
file1=open(r'C:\Users\pc\Desktop\input1.txt','r', encoding="utf-8")
lines= [] 
x=[]
count=0
for paragraph in file1:
    for char in ',![]@()+:':
        paragraph=paragraph.replace(char,'')
    lines = str.split(paragraph,'.')
    for each_line in lines:
        if each_line.find("며여용")>0:
            count+=1
        else:
            client_id="ebTOYs9YiY0RIEV8Vn8u"
            client_secret="gSw_0r8zLc"
            encText=urllib.parse.quote(each_line)
            client_id="ebTOYs9YiY0RIEV8Vn8u"
            client_secret="gSw_0r8zLc"
            encText=urllib.parse.quote(each_line)
            data="speaker=mijin&speed=0&text="+encText;
            url="https://openapi.naver.com/v1/voice/tts.bin"
            request=urllib.request.Request(url)
            request.add_header("X-Naver-Client-Id",client_id)
            request.add_header("X-Naver-Client-Secret",client_secret)
            response = urllib.request.urlopen(request,data=data.encode('utf-8'))
            rescode = response.getcode()
            if (rescode==200):
                print("TTS mp3 저장")
                response_body = response.read()
                f=open('C:\\Users\\pc\\Desktop\\lll\\'+str(i)+'.mp3', 'wb')
                f.write(response_body)
            else:
                print("Error Code:" + rescode)
file1.close()
                
