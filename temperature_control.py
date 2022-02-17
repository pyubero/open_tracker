# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:44:08 2022

@author: logslab
"""


import numpy as np
from time import sleep
from pytracker import myArduTracker
from datetime import datetime

DT        =  5
T_OBJ     = 20
HIST_SIZE = 10


# Find devices connected to computer
devices = myArduTracker.find_devices()
print('These are the devices found:')
_ = [ print('-- %s' % dev) for dev in devices ]
print('')


# Connect to last device
# ... status led should blink in orange
board = myArduTracker.myArduTracker( 'COM5' )




history = np.zeros((HIST_SIZE,))



def update_history(board, history):
    new_value = board.temperature()
    return np.array( [new_value, *history[:-1]]  )
    
    
    
def update_temperature(board, history):
    tobj = 20
    
    
    error = history-tobj
    output= 75 + Dpwm
    print(datetime.now(), np.round(output,2), np.round(history[0],2) )
    board.tec( int(output) )


# Turn ON the cooling system
board.fan(1)
tStart = datetime.now()

for _ in range(300):
    history = update_history( board, history )
    update_temperature(board, history)
    
    sleep(1)


























