# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 15:35:35 2022

@author: logslab
"""

import os
import numpy as np
from time import sleep
from pytracker import myArduTracker
from datetime import datetime

DT     =  60
FNAME  = './logs_temperature_record.txt' 

# Find devices connected to computer
devices = myArduTracker.find_devices()
print('These are the devices found:')
_ = [ print('-- %s' % dev) for dev in devices ]
print('')


# Connect to last device
# ... status led should blink in orange
board = myArduTracker.myArduTracker( devices[-1] )
new_value = board.temperature()
sleep(1)

# Clear file
with open(FNAME,'w') as file:
    pass

while True:
    
    new_value = board.temperature()
    output    = '%1.2f\t%s' % (new_value, datetime.now().strftime('%H:%M'))
    with open(FNAME,'a+') as file:
        file.write( output+'\n' )
    
    print(output, datetime.now().strftime('%H:%M'))    
    sleep(900)







