# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 17:43:45 2022

@author: logslab
"""
import numpy as np
from time import sleep
from pytracker import myArduTracker

TEST_LED1 = True
TEST_LED2 = True
TEST_LED3 = True
TEST_TEMP = True
TEST_POT  = True







# Find devices connected to computer
devices = myArduTracker.find_devices()
print('These are the devices found:')
_ = [ print('-- %s' % dev) for dev in devices ]
print('')

# Connect to last device
# ... status led should blink in orange
board = myArduTracker.myArduTracker( devices[-1] )


if TEST_LED1:
    print('Sweeping led1...')
    for jj in np.arange(0,256,15):
        board.led1( int(jj) )
        sleep(0.2)
        
    for jj in np.arange(255,-1,-15):
        board.led1( int(jj) )
        sleep(0.2)


if TEST_LED2:
    print('Sweeping led2...')
    for jj in np.arange(0,256,15):
        board.led2( int(jj) )
        sleep(0.2)
        
    for jj in np.arange(255,0-1,-15):
        board.led2( int(jj) )
        sleep(0.2)


if TEST_LED3:
    print('Sweeping led3...')
    for jj in np.arange(0,256,15):
        board.led3( int(jj) )
        sleep(0.2)
        
    for jj in np.arange(255,-1,-15):
        board.led3( int(jj) )
        sleep(0.2)



if TEST_POT:
    print('Reading pot values...')
    for _ in range(20):
        print('\b'*30, end='')
        print('Pot: %1.3f' % board.pot() , end='')
        sleep(0.2)
    print('')
        
        
if TEST_TEMP:
    print('Reading temperature...')
    for _ in range(20):
        print('\b'*30,'T: %1.2f ºC' % board.temperature() , end='')
        sleep(0.2)       
    print('')



board.close()
        