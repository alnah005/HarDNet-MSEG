# -*- coding: utf-8 -*-
"""
file: randomSplitter.py

@author: Suhail.Alnahari

@description: 

@created: 2021-03-06T17:33:37.847Z-06:00

@last-modified: 2021-03-06T17:49:47.930Z-06:00
"""

# standard library
# 3rd party packages
# local source

import os
from random import shuffle
import shutil

files1 = os.listdir(os.path.join("haveLabel","images"))
shuffle(files1)
PERCENTAGE = 0.2

for i in range(int(len(files1)*PERCENTAGE)):
    shutil.move(os.path.join("haveLabel","images",files1[i]),os.path.join("haveLabel_val","images",files1[i]))
    shutil.move(os.path.join("haveLabel","masks",files1[i].split(".jpg")[0]+"_mask.jpg"),os.path.join("haveLabel_val","masks",files1[i].split(".jpg")[0]+"_mask.jpg"))