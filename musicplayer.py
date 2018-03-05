# -*- coding: utf-8 -*-
#!/usr/bin/python
# -*- coding: cp949 -*-

"""
Created on Mon Feb  5 09:19:26 2018

@author: pc
"""


from pygame import mixer
import pygame
import time

pygame.init()
#파일로드
for i in range(total_num):
   pygame.mixer.music.load('C:\\Users\\pc\\Desktop\\isthe\\'+str(i)+'.mp3')
   mixer.music.play()
   time.sleep(60)

pygame.mixer.music.stop()