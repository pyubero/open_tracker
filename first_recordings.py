# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:08:07 2022

@author: logslab
"""




from vimba import *
from pytracker import myArduTracker
from pytracker import myCamera_Alvium as myCamera



# # Find devices connected to computer
# devices = myArduTracker.find_devices()
# print('These are the devices found:')
# _ = [ print('-- %s' % dev) for dev in devices ]
# print('')

# # Connect to last device
# # ... status led should blink in orange
# print('Connecting to last device...', end='')
# board = myArduTracker.myArduTracker( devices[-1] )
# print(' done.')

# # Connect potentiometer, 
# print('Linking potentiometer to led1...', end='')
# board.start_pot_link()
# print(' done.')



# Create myCamera object
cam = myCamera.myCamera(0)

# Set some properties
cam.set('width', 2592)
cam.set('height', 1944)
#...
cam.recording_filename = 'video_NVIDEO.avi' #... NVIDEO will be substituted by video index
cam.recording_format   = 'MJPG'
cam.recording_fps      = 1
cam.recording_totaltime= 3600  #... in seconds
cam.recording_maxtime  = 2000  #... in seconds



# Retrieve some properties
print('Resolution: %dx%d' % ( cam.get('width'), cam.get('height')) )


# Start stream, and preview
#... and start recording by pressing R
cam.start_preview( formfactor=0.33)



# Close camera
cam.close()








































