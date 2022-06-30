# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:08:07 2022

@author: logslab
"""




# from vimba import *
from pytracker import myCamera_Alvium as myCamera

# from time import sleep
# from multiprocessing import Process
# from pytracker import myArduTracker
# from pytracker import myArduinoGui as GUI


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
# # print('Linking potentiometer...', end='')
# # board.start_pot_link('led2')
# board.led2(60)
# print(' done.')

# sleep(2)
# p = Process(target= GUI.run() )
# p.start()
# GUI.run( blocking=False )

# Create myCamera object
cam = myCamera.myCamera(0)

# Set some properties
cam.set('width', 2592)
cam.set('height', 1944)
cam.set('exposure', 36964) # good values are 1775, 3563, 4832, 5597, 7395


frame = cam.snapshot()
cam.summary()

#...

cam.recording_filename = './videos/no_cond_2h_TDC_1_sin_food_conNaCl_DATETIME_NVIDEO.avi' #... NVIDEO will be substituted by video index
cam.recording_format   = 'MP42'
cam.recording_fps      = 2
cam.recording_totaltime=  1.5*3600  #... in seconds
cam.recording_maxtime  = 9999999999  #... in seconds



# Retrieve some properties
print('Resolution: %dx%d' % ( cam.get('width'), cam.get('height')) )


# Start stream, and preview
#... and start recording by pressing R
if cam.start_streaming():
    cam.start_preview( formfactor=0.33)



# Close camera
cam.close()
# board.close()








