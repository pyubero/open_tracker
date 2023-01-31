# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:08:07 2022

@author: logslab
"""
from pytracker import myCamera_Alvium as myCamera
#from pytracker import myCamera_UVC as myCamera

SHUTDOWN_WHEN_FINISHED = False


# Create myCamera object
cam = myCamera.myCamera(0)


# Set some properties
cam.set('width', 2592)
cam.set('height', 1944)
cam.set('exposure', 36964) # good values are 1775, 3563, 4832, 5597, 7395, 36964


# Initialize camera
frame = cam.snapshot()
cam.summary()


# Set more properties
cam.recording_filename = './videos/video_DATETIME_NVIDEO.avi' #... NVIDEO will be substituted by video index
cam.recording_format   = 'MP42'
cam.recording_fps      = 2
cam.recording_totaltime= 2*3600  #... in seconds
cam.recording_maxtime  = 0.5*3600  #... in seconds


# Retrieve some properties
print('Resolution: %dx%d' % ( cam.get('width'), cam.get('height')) )


# Start stream, and preview
#... and start recording by pressing R
if cam.start_streaming():
    cam.start_preview( formfactor=0.3 )


# Close camera
cam.close()


# Shut Down PC
if SHUT_DOWN_WHEN_FINISHED:
    print('The computer will shut down in 60s.')
    print('Press ctrl+C to cancel this action.')
    
    for _ in tqdm(range(60)):
        sleep(1)
    os.system("shutdown /s /t 1")
