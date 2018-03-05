# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:38:00 2018

@author: pc
"""


with open(r'C:\Users\pc\Desktop\hip\data\hiphop.txt', 'r', encoding = 'utf-8') as f:
    hiphop_txt = f.read()
tw = Twitter()
stop_words = open(r'C:\Users\pc\Desktop\hip\data\hiphop.txt', 'r',encoding = 'utf-8').read().split('\n')

def normalize(lyric):
    nouns = tw.nouns(lyric)
    lyric_noun = [ noun for noun in nouns if len(noun) > 1 and noun not in stop_words]
    return lyric_noun
normalized_text = normalize(hiphop_txt)
WC = WordCloud(r'C:\Windows\Fonts\font\Daum_SemiBold.ttf', width=700, height=700)
hiphop_wc = WC.generate(' '.join(normalized_text))
hiphop_wc.to_file(r'C:\Users\pc\Desktop\hip\data\hiphopwordcloud.png')

wordcount={}
for i in range(len(normalized_text)):
    number=normalized_text.count(normalized_text[i])
    wordcount[number]=wordcount.get(number,'')+' '+normalized_text[i] # 갯수 : 단어 형태의 딕셔너리로 저장
