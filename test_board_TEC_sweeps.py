# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 17:43:45 2022

@author: logslab
"""
import numpy as np
from time import sleep
from pytracker import myArduTracker
from datetime import datetime


DTIME    =   5  # in seconds
TIME_MAX = 300 # in seconds
DTEC     =  25
TEC_MIN  =   0
TEC_MAX  = 151
EXPORT   = False
FILENAME = 'tec_test.csv'



# Find devices connected to computer
devices = myArduTracker.find_devices()
print('These are the devices found:')
_ = [ print('-- %s' % dev) for dev in devices ]
print('')


# Connect to last device
# ... status led should blink in orange
board = myArduTracker.myArduTracker( 'COM5' )




# Create empty file to export data
if EXPORT:
    with open(FILENAME,'w+') as file:
        pass


# Turn ON the cooling system
board.tec(0)
board.fan(1)
tStart = datetime.now()


# Start loop - cooling
for tec_pwm in np.arange(TEC_MIN,TEC_MAX,DTEC):
    
    board.tec( int(tec_pwm), mode='cool' )
    print('\n Starting with PWM=%d' % tec_pwm)
    
    for _ in range( int(TIME_MAX/DTIME)):
        sleep(DTIME)
        temp = board.temperature()
        tnow = (datetime.now()-tStart).total_seconds()
        print('\b'*30,'[%4d] T: %1.2f ºC' % (tnow,temp) , end='')
           
        if EXPORT:
            with open(FILENAME,'a') as file:
                file.write('%1.1f;%d;%1.2f\n' % (tnow, tec_pwm, temp)  )


# Start loop - heating
for tec_pwm in np.arange(TEC_MIN,TEC_MAX,DTEC):
    
    board.tec( int(tec_pwm), mode='heat' )
    print('\n Starting with PWM=%d' % tec_pwm)
    
    for _ in range( int(TIME_MAX/DTIME)):
        sleep(DTIME)
        temp = board.temperature()
        tnow = (datetime.now()-tStart).total_seconds()
        print('\b'*30,'[%4d] T: %1.2f ºC' % (tnow,temp) , end='')
           
        if EXPORT:
            with open(FILENAME,'a') as file:
                file.write('%1.1f;%d;%1.2f\n' % (tnow, tec_pwm, temp)  )





print('Cooling a bit before turning off...')
board.tec(0)
board.fan(1)
sleep(10)
board.close()
        