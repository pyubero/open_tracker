# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:48:56 2022

@author: logslab
"""


from pytracker import myArduTracker

if __name__ == '__main__':
    # Find devices connected to computer
    devices = myArduTracker.find_arduinos()
    
    print('These are the devices found:')
    _ = [ print('-- %s' % dev) for dev in devices ]
    print('')
    
    # Connect to last device
    # ... status led should blink in orange
    print('Connecting to last device...', end='')
    board = myArduTracker.myArduTracker( devices[-1] )
    print(' done.')
    
    # Connect potentiometer, 
    # print('Linking potentiometer...', end='')
    # board.start_pot_link('led2')
    board.led2(0)
    board.led3(5)
    board.write(13,1)
    
    print(' done.')
    
    board.close( digital=[] )



















