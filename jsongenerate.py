# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 20:04:32 2018

@author: pc
"""

import json
from collections import OrderedDict

file_data = OrderedDict()
file_data[""]=" "

print(json.dumps(file_data, ensure_ascii=False, indent="\t"))

with open(r'C:\Users\pc\Desktop\shm\tacotron\datasets\yuinna\alignment.json','w',encoding='utf-8') as make_file:
    json.dump(file_data,make_file,ensure_ascii=False,indent='\t')