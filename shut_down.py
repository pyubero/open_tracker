# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 18:02:13 2022

@author: logslab
"""

import os
from time import sleep
from tqdm import tqdm


for _ in tqdm(range(1800)):
    sleep(4)
os.system("shutdown /s /t 1")
