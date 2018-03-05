# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:24:40 2018

@author: korea
"""
#mysql -u root -p 
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import codecs
import pandas as pd
import konlpy
from konlpy.tag import Twitter

import gensim
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
from gensim import corpora, models
from gensim.models import LdaMulticore
from gensim.models import Word2Vec
from gensim.corpora import Dictionary, MmCorpus

import pyLDAvis
import pyLDAvis.gensim
import pickle

from wordcloud import WordCloud
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import mysql.connector
from os import path
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, value
from selenium.common.exceptions import NoSuchElementException
'''
def replaceInFile(file_path, old, newstr):
    f = codecs.open(file_path, 'r', encoding='utf8') 
    read_file = f.read() 
    f.close()
    new_file = codecs.open(file_path,'w', encoding='utf8') 
    for line in read_file.split("\n"): 
        new_file.write(line.replace(old, newstr)) 
        new_file.write("\n") #close file new_file.close()
        #close file 
        new_file.close()
'''
        
driver = webdriver.Chrome(r'C:\Users\pc\Desktop\chromedriver.exe')
driver.get("http://www.melon.com/genre/song_list.htm?gnrCode=GN0200")

lyric_page = driver.find_element_by_xpath("//a[@class='btn button_icons type03 song_info']")
lyric_page.click()


driver = webdriver.Chrome()
for a in range(10):
    a = a * 50 + 1
    url = "http://www.melon.com/genre/song_list.htm?gnrCode=GN0300#params%5BgnrCode%5D=GN0300&params%5BdtlGnrCode%5D=&params%5BorderBy%5D=NEW&po=pageObj&startIndex={}".format(a)
    driver.get(url)
    for i in range(50):
        try:
            element = driver.find_elements_by_xpath("//a[@class='btn button_icons type03 song_info']")
            element[i].click()
            title = driver.find_element_by_xpath("//div[@class='song_name']")
            likes = driver.find_element_by_xpath("//span[@id='d_like_count']")
            singer = driver.find_element_by_xpath("//div[@class='artist']")
            lyrics = driver.find_element_by_xpath("//div[@class='lyric']")
            print ("가사 :")
            print (lyrics.text)
            with open(r'C:\Users\pc\Desktop\hip\data\hiphop.txt', 'a+', encoding = 'utf-8') as f:
                hiphop_txt=f.write(lyrics.text)
            try:
                query = """INSERT INTO lyrics
                VALUES ('{}', '{}', '{}', '{}')""" \
               .format(singer.text.replace("'", '').replace('"', ''), title.text.replace("'", '').replace('"', ''), int(likes.text.replace(',', '')), lyrics.text.replace(".", '').replace("*", '').replace("-", '').replace(",", '').replace("'", '').replace('"', ''))
                cursor.execute(query)
                db.commit()
            except Exception as e:
                db.rollback()
                print (e)
        except:
            driver.back()
            i += 1      
    driver.back()
    print(a)
    a += 1


print("DONE")

db.close()