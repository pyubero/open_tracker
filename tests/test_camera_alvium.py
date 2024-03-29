# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:56:20 2022

@author: logslab
"""

import cv2
import numpy as np
from vimba import *
from datetime import datetime
from time import sleep
from matplotlib import pyplot as plt
from tqdm import tqdm
from pytracker import myCamera_Alvium as myCamera


FILENAME = 'video_alvium.avi'
FORMAT   = 'MJPG'
LENGTH   = 10 #... in seconds
FPS      = 10

# Create myCamera object
cam = myCamera.myCamera(0)
cam.summary()

# Set some properties
cam.set('width', 2592)
cam.set('height', 1944)
cam.set('exposure', 7385) #1270,1775,3563,4832,5597,7385


# Retrieve some properties
print('Resolution: %dx%d' % ( cam.get('width'), cam.get('height')) )

# Take a picture
frame = cam.snapshot()
#plt.imshow(frame)
cam.summary()


# Starting streaming (or not, as you wish)
cam.start_streaming()   
 

# Start stream, and preview
#... and start recording by pressing R
cam.start_preview( formfactor=0.33)

# Take a picture and save it
frame = cam.snapshot()
cv2.imwrite( './test_video/snapshot.png', frame)



# Close camera
cam.close()







