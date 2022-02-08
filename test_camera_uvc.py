# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:56:20 2022

@author: logslab
"""

import cv2
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from pytracker import myCamera_UVC as myCamera


    
# Create myCamera object, which automatically opens it 
cam = myCamera.myCamera(0)

# Set some properties
cam.set('resolution', (3000,3000) )
cam.set('brightness',16)

# Retrieve some properties
print('Resolution: %dx%d' % ( cam.get('width'), cam.get('height')) )
print('Brightness is %d' % cam.get('brightness') )

# Take a picture
cv2.imshow( 'window' , cam.snapshot(formfactor = 0.5) )
cv2.waitKey(0)
cv2.destroyAllWindows()

# Start stream, and preview
#... and start recording by pressing R
cam.start_preview( formfactor=0.5)

# Close camera
cam.close()





